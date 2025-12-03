"""
Full end-to-end optimized transaction reconciliation pipeline
- Hybrid CNN+Transformer character encoder for string fields
- SignedAmount included inside neural encoder
- Contrastive self-supervised pretraining (positive pairs from MatchGroupId)
- DBSCAN clustering (with simple automatic hyperparameter search)
- Greedy CR/DR amount balancing per cluster (fast, no graph matching)
- Optional evaluation vs MatchGroupId for precision/recall/F1
"""

import random
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128
CNN_CHAR_EMBED = 16
CNN_OUT_CH = 64
SEQ_LEN = 64
FIELDS = ["TransactionRefNo","MerchantRefNum","AcquireRefNumber","WebOrderNumber","PONumber","CardNo"]
AMOUNT_ABS_TOL = 0.01
AMOUNT_REL_TOL = 0.01
MAX_MULTIWAY = 4

DBSCAN_EPS = [0.2,0.5,0.8,1.0]
DBSCAN_MIN_SAMPLES = [2,3,5]

# -----------------------
# 0) Load / sample data
# -----------------------
data = {
    "DocumentDate": ["02/01/2025","02/01/2025"],
    "DocType": ["unknown","unknown"],
    "TransactionType": ["SAP","SAP"],
    "BankTrfRef": ["unknown","unknown"],
    "Amount": [7800,-7800],
    "TransactionRefNo": ["W1dsfsafjdjfb","W1dsfsafjdjfb"],
    "MerchantRefNum": ["777741344598","777741344598"],
    "CR_DR": ["CR","DR"],
    "GLRecordID": ["unknown","unknown"],
    "OID": ["unknown","unknown"],
    "PONumber": ["13u350u5u05","13u350u5u05"],
    "CardNo": ["13415555531535","13415555531535"],
    "AccountingDocNum": ["KD0nfdkk","KD0nfdkk"],
    "AcquireRefNumber": ["unknown","unknown"],
    "WebOrderNumber": ["W1342421414","W1342421414"],
    "MatchGroupId": ["14443553","14443553"],
    "Source": ["SAP","SAP"],
    "SourceType": ["Internal","Internal"]
}
df = pd.DataFrame(data)
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# SignedAmount: CR=+, DR=-
df["CR_DR"] = df["CR_DR"].fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR":1.0,"DR":-1.0}).fillna(1.0)

# Label encoding for categorical columns
label_cols = [c for c in ["DocType","TransactionType","Source","SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col+"_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------
# 1) Build char vocabulary
# -----------------------
all_text = ""
for f in FIELDS:
    all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars)==0:
    chars=list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c:i+1 for i,c in enumerate(chars)}  # 0 for padding
IDX_PAD = 0
VOCAB_SIZE = len(char_to_idx)+1

def encode_text(s, max_len=SEQ_LEN):
    s="" if s is None else str(s)
    idxs=[char_to_idx.get(ch,0) for ch in s[:max_len]]
    if len(idxs)<max_len: idxs+=[IDX_PAD]*(max_len-len(idxs))
    return idxs

def build_row_seq(row):
    seq=[]
    for f in FIELDS:
        seq+=encode_text(row.get(f,""),SEQ_LEN)
    return seq

SEQ_TOTAL = SEQ_LEN*len(FIELDS)
sequences = np.array([build_row_seq(r) for _,r in df.iterrows()],dtype=np.int64)

# -----------------------
# 2) Dataset for contrastive learning using MatchGroupId positives
# -----------------------
class MatchGroupContrastiveDataset(Dataset):
    def __init__(self, df):
        self.df=df.reset_index(drop=True)
        self.groups=defaultdict(list)
        for idx, mg in enumerate(self.df["MatchGroupId"].astype(str)):
            self.groups[mg].append(idx)
        self.anchors=[i for mg, idxs in self.groups.items() for i in idxs if len(idxs)>=2]
    def __len__(self):
        return len(self.anchors)
    def __getitem__(self, idx):
        anchor_idx = self.anchors[idx]
        mg = str(self.df.loc[anchor_idx,"MatchGroupId"])
        members=self.groups[mg]
        pos = anchor_idx
        while pos==anchor_idx:
            pos=random.choice(members)
        x1 = np.array(build_row_seq(self.df.loc[anchor_idx]),dtype=np.int64)
        x2 = np.array(build_row_seq(self.df.loc[pos]),dtype=np.int64)
        a1 = float(self.df.loc[anchor_idx,"SignedAmount"])
        a2 = float(self.df.loc[pos,"SignedAmount"])
        return x1,x2,a1,a2

dataset = MatchGroupContrastiveDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE,len(dataset)), shuffle=True, drop_last=False)

# -----------------------
# 3) Model: Per-field CNN + Transformer + SignedAmount embedding
# -----------------------
class PerFieldCNN(nn.Module):
    def __init__(self,vocab_size,char_emb=CNN_CHAR_EMBED,conv_out=CNN_OUT_CH,k=5):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,char_emb,padding_idx=IDX_PAD)
        self.conv=nn.Conv1d(char_emb,conv_out,kernel_size=k,padding=k//2)
        self.pool=nn.AdaptiveMaxPool1d(1)
    def forward(self,seq):
        x=self.embed(seq).transpose(1,2)
        x=F.relu(self.conv(x))
        x=self.pool(x).squeeze(-1)
        return x

class HybridEncoderWithAmount(nn.Module):
    def __init__(self,vocab_size,seq_per_field=SEQ_LEN,num_fields=len(FIELDS),
                 char_emb=CNN_CHAR_EMBED,conv_out=CNN_OUT_CH,trans_dim=128,nhead=4,num_layers=2,proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields=num_fields
        self.seq_per_field=seq_per_field
        self.field_encoder=PerFieldCNN(vocab_size,char_emb=char_emb,conv_out=conv_out)
        encoder_layer=nn.TransformerEncoderLayer(d_model=conv_out,nhead=nhead,dim_feedforward=trans_dim,activation='relu')
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.amount_project=nn.Sequential(nn.Linear(1,conv_out),nn.ReLU())
        self.proj=nn.Sequential(nn.Linear(conv_out,conv_out),nn.ReLU(),nn.Linear(conv_out,proj_dim))
    def forward(self,seq_batch,amount_batch):
        b=seq_batch.size(0)
        x=seq_batch.view(b,self.num_fields,self.seq_per_field)
        field_vecs=[]
        for f in range(self.num_fields):
            seq_f=x[:,f,:].long()
            v=self.field_encoder(seq_f)
            field_vecs.append(v.unsqueeze(1))
        field_stack=torch.cat(field_vecs,dim=1)
        tr_in=field_stack.transpose(0,1)
        tr_out=self.transformer(tr_in)
        pooled=tr_out.mean(dim=0)
        amt_proj=self.amount_project(amount_batch.view(-1,1))
        combined=pooled+amt_proj
        z=self.proj(combined)
        z=F.normalize(z,p=2,dim=1)
        return z

model=HybridEncoderWithAmount(VOCAB_SIZE).to(DEVICE)

# -----------------------
# 4) InfoNCE loss
# -----------------------
def nt_xent_loss_from_pair_embeddings(z1,z2,temperature=TEMPERATURE):
    B=z1.size(0)
    z=torch.cat([z1,z2],dim=0)
    sim=torch.matmul(z,z.T)/temperature
    mask=(~torch.eye(2*B,2*B,dtype=torch.bool,device=DEVICE)).float()
    positives=torch.cat([torch.diag(sim,B),torch.diag(sim,-B)],dim=0)
    nom=torch.exp(positives)
    denom=(mask*torch.exp(sim)).sum(dim=1)
    loss=-torch.log(nom/denom)
    return loss.mean()

optimizer=torch.optim.Adam(model.parameters(),lr=LR)

# -----------------------
# 5) Normalize amounts
# -----------------------
amounts=df["SignedAmount"].values.reshape(-1,1)
amt_scaler=StandardScaler()
if len(amounts)>1: amt_scaler.fit(amounts)
def scale_amounts(x):
    x=np.array(x).reshape(-1,1).astype(float)
    return (x-amt_scaler.mean_)/(amt_scaler.scale_+1e-9)

# -----------------------
# 6) Train contrastive encoder
# -----------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss=0.0
    n=0
    for x1_np,x2_np,a1,a2 in loader:
        x1=torch.tensor(x1_np,dtype=torch.long,device=DEVICE)
        x2=torch.tensor(x2_np,dtype=torch.long,device=DEVICE)
        a1_scaled=torch.tensor(scale_amounts(a1),dtype=torch.float32,device=DEVICE).squeeze(1)
        a2_scaled=torch.tensor(scale_amounts(a2),dtype=torch.float32,device=DEVICE).squeeze(1)
        optimizer.zero_grad()
        z1=model(x1,a1_scaled)
        z2=model(x2,a2_scaled)
        loss=nt_xent_loss_from_pair_embeddings(z1,z2)
        loss.backward()
        optimizer.step()
        batch_n=x1.size(0)
        epoch_loss+=loss.item()*batch_n
        n+=batch_n
    if (epoch+1)%5==0 or epoch==0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss/max(1,n):.6f}")

# -----------------------
# 7) Generate embeddings
# -----------------------
model.eval()
with torch.no_grad():
    seq_tensor=torch.tensor(sequences,dtype=torch.long,device=DEVICE)
    amt_scaled=torch.tensor(scale_amounts(df["SignedAmount"].values),dtype=torch.float32,device=DEVICE).squeeze(1)
    embeddings=model(seq_tensor,amt_scaled).cpu().numpy()

num_feats_scaled=(df[["SignedAmount"]].values.astype(float)-amt_scaler.mean_)/(amt_scaler.scale_+1e-9)
cat_feats=np.array([df[c+"_enc"].values for c in label_cols]).T if label_cols else np.zeros((len(df),0))
X=np.hstack([embeddings,num_feats_scaled,cat_feats])

# -----------------------
# 8) DBSCAN clustering with simple hyperparameter search
# -----------------------
best_score=-1.0
best_labels=None
best_params=None
if len(X)<2:
    best_labels=np.array([-1]*len(X))
    best_params=(None,None)
else:
    for eps in DBSCAN_EPS:
        for ms in DBSCAN_MIN_SAMPLES:
            clusterer=DBSCAN(eps=eps,min_samples=ms,metric='euclidean')
            labels=clusterer.fit_predict(X)
            mask=labels!=-1
            if mask.sum()<2: score=-1.0
            else: score=silhouette_score(X[mask],labels[mask])
            if score>best_score:
                best_score=score
                best_labels=labels
                best_params=(eps,ms)

df["raw_cluster"]=best_labels
print("DBSCAN best params:",best_params,"best silhouette:",best_score)

# -----------------------
# 9) Greedy matching for CR/DR balancing
# -----------------------
n=len(df)
final_cluster_map={i:-1 for i in range(n)}
next_cid=0

def greedy_pairing(pos_list,neg_list):
    pos_sorted=sorted(pos_list,key=lambda x:-abs(x[1]))
    neg_sorted=sorted(neg_list,key=lambda x:-abs(x[1]))
    neg_available=neg_sorted.copy()
    pairs=[]
    TOP_K=50
    for i_idx,i_amt in pos_sorted:
        if not neg_available: break
        best_j=None
        best_diff=float("inf")
        limit=min(TOP_K,len(neg_available))
        for j_idx,j_amt in neg_available[:limit]:
            diff=abs(i_amt+j_amt)
            if diff<best_diff:
                best_diff=diff
                best_j=(j_idx,j_amt)
                if best_diff<=AMOUNT_ABS_TOL: break
        if best_j is None: continue
        rel_tol=AMOUNT_REL_TOL*max(abs(i_amt),abs(best_j[1]),1.0)
        if best_diff<=max(AMOUNT_ABS_TOL,rel_tol):
            pairs.append((i_idx,best_j[0]))
            neg_available.remove(best_j)
    paired_pos={p for p,q in pairs}
    paired_neg={q for p,q in pairs}
    leftovers_pos=[i for i,a in pos_sorted if i not in paired_pos]
    leftovers_neg=[j for j,a in neg_sorted if j not in paired_neg]
    return pairs,leftovers_pos,leftovers_neg

def process_cluster_members(members):
    global next_cid
    pos=[(i,df.loc[i,"SignedAmount"]) for i in members if df.loc[i,"SignedAmount"]>0]
    neg=[(i,df.loc[i,"SignedAmount"]) for i in members if df.loc[i,"SignedAmount"]<0]
    if not pos or not neg: return
    pairs,left_pos,left_neg=greedy_pairing(pos,neg)
    for i,j in pairs:
        final_cluster_map[i]=next_cid
        final_cluster_map[j]=next_cid
        next_cid+=1
    leftovers=left_pos+left_neg
    if 2<=len(leftovers)<=MAX_MULTIWAY:
        assigned=set()
        for r in range(min(MAX_MULTIWAY,len(leftovers)),2,-1):
            for combo in combinations(leftovers,r):
                if any(c in assigned for c in combo): continue
                s=sum(df.loc[c,"SignedAmount"] for c in combo)
                if abs(s)<=AMOUNT_ABS_TOL:
                    for c in combo:
                        final_cluster_map[c]=next_cid
                        assigned.add(c)
                    next_cid+=1

unique_labels=sorted(set(best_labels))
raw_clusters={lbl:[i for i,lab in enumerate(best_labels) if lab==lbl] for lbl in unique_labels}
for lbl,members in raw_clusters.items():
    if lbl==-1: continue
    process_cluster_members(members)

# global greedy pass for remaining unassigned
unassigned=[i for i in range(n) if final_cluster_map[i]==-1]
if unassigned:
    pos_un=[(i,df.loc[i,"SignedAmount"]) for i in unassigned if df.loc[i,"SignedAmount"]>0]
    neg_un=[(i,df.loc[i,"SignedAmount"]) for i in unassigned if df.loc[i,"SignedAmount"]<0]
    pairs,left_pos,left_neg=greedy_pairing(pos_un,neg_un)
    for i,j in pairs:
        final_cluster_map[i]=next_cid
        final_cluster_map[j]=next_cid
        next_cid+=1
    remaining=[i for i in unassigned if final_cluster_map[i]==-1]
    if 2<=len(remaining)<=MAX_MULTIWAY:
        assigned=set()
        for r in range(min(MAX_MULTIWAY,len(remaining)),2,-1):
            for combo in combinations(remaining,r):
                if any(c in assigned for c in combo): continue
                s=sum(df.loc[c,"SignedAmount"] for c in combo)
                if abs(s)<=AMOUNT_ABS_TOL:
                    for c in combo:
                        final_cluster_map[c]=next_cid
                        assigned.add(c)
                    next_cid+=1

df["FinalCluster"]=[final_cluster_map.get(i,-1) for i in range(n)]

# -----------------------
# 10) Optional evaluation
# -----------------------
if "MatchGroupId" in df.columns:
    true_pairs=set()
    pred_pairs=set()
    for i,j in combinations(range(n),2):
        if str(df.loc[i,"MatchGroupId"])==str(df.loc[j,"MatchGroupId"]):
            true_pairs.add((i,j))
        if df.loc[i,"FinalCluster"]!=-1 and df.loc[i,"FinalCluster"]==df.loc[j,"FinalCluster"]:
            pred_pairs.add((i,j))
    tp=len(true_pairs & pred_pairs)
    fp=len(pred_pairs - true_pairs)
    fn=len(true_pairs - pred_pairs)
    prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    print(f"Evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# -----------------------
# 11) Output
# -----------------------
print("Sample output:")
show_cols=["SignedAmount","raw_cluster","FinalCluster"]+([c+"_enc" for c in label_cols] if label_cols else [])
print(df[show_cols].head())
df.to_csv("clusters_hybrid_contrastive_dbscan_greedy.csv",index=False)
print("Done.")
