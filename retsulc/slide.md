Use this framing throughout: **all numbers marked “Illustrative” are believable placeholders for discussion only, not Yahoo, AMD, or customer results. Replace them with approved internal measurements before presenting.** Keep the resume-reported 50%+ GPU cost reduction and 45%+ latency improvement distinct from illustrative examples.

## 1. Production AI Workloads: What Hardware Benchmarks Miss

**On-slide content**

- Production AI performance spans the model, runtime, serving stack, request mix, and GPU fleet.
- Goal: connect real application behavior to benchmarking decisions.
- Scope: LLM serving, workload characterization, agentic workflows, and production capacity planning.

**Visual placeholder**

`[Hero visual: user request flowing through gateway → model runtime → GPU fleet → tools/services → response]`

**Speaker example**

> A single “tokens/sec” number is useful, but it cannot explain whether a user received a fast, reliable response during a traffic burst.

---

## 2. Talk Structure and Evidence Boundaries

**On-slide content**

| Section | Question |
|---|---|
| Workload evolution | What changed in AI serving? |
| Production experience | What patterns appeared in real systems? |
| Benchmarking | What should we measure? |
| AMD discussion | Which workload profiles should guide future evaluation? |

**Footer**

> Facts labeled “Resume-supported” come from my resume. “Illustrative” data and diagrams show realistic example scenarios only.

**Visual placeholder**

`[Simple four-step agenda bar]`

---

## 3. AI Workload Evolution

**On-slide content**

| Workload era | Typical request | Primary performance focus |
|---|---|---|
| Traditional ML | One prediction | Latency, throughput |
| Deep learning | Batch inference | Compute efficiency, memory bandwidth |
| Generative AI | Prompt + streamed response | TTFT, tokens/sec, KV-cache capacity |
| Agentic AI | Multi-step task + tools | End-to-end latency, reliability, cost per completed task |

**Illustrative example**

- A recommendation request may use one model call and complete in **<100 ms**.
- A chat request may require **1–4 seconds** before the first token, then stream output.
- A tool-using workflow may make **3–8 model calls** plus external service calls before completion.

**Visual placeholder**

`[Horizontal evolution timeline with request shape growing from one model call to multi-step workflow]`

---

## 4. From Model Metrics to Production Outcomes

**On-slide content**

**Model/runtime metrics**
- TTFT
- Tokens/sec
- P50/P95 latency
- Memory footprint
- GPU utilization

**Production-service metrics**
- Queue depth and wait time
- Concurrency supported per replica
- Error rate, including 429s
- Cost per request or completed task
- Capacity headroom during peak demand

**Illustrative example**

| Metric | Isolated benchmark | Production observation |
|---|---:|---:|
| GPU utilization | 88% | 55–75% |
| P95 TTFT | 450 ms | 1.8 s |
| Error rate | 0% | 0.7% during burst |
| Request mix | Fixed prompt | Short, long, retries, tool calls |

**Takeaway**

> A high-throughput isolated benchmark may not predict user-visible latency under variable production traffic.

**Visual placeholder**

`[Two-layer diagram: kernel/model metrics beneath service-level outcomes]`

---

## 5. Production Experience at a Glance

**On-slide content**

- **Resume-supported:** Yahoo Mail Intelligence work for a unit serving **1B+ daily traffic**.
- **Resume-supported:** LLM platform experience serving **220M+ users** and **billions of requests/day**, stated as roughly **10K to 50K QPS**.
- **Resume-supported:** Production work across GCP model serving, GPU fleet sizing, LLM gateways, RAG, agentic workflows, and observability.
- **Resume-supported:** Earlier enterprise MLOps/RAG experience at Metrix and production NLP/ML work at Pearson.

**Visual placeholder**

`[Timeline: Pearson → Metrix enterprise ML platforms → Yahoo production LLM systems]`

**Optional callout**

> The common thread: model quality matters, but operating behavior determines whether a system scales.

---

## 6. Yahoo Production Inference Architecture

**On-slide content**

- **Resume-supported:** End-to-end architecture covered application, model, GPU fleet, and on-device llama.cpp layers.
- **Resume-supported:** Fine-tuned Qwen variants, Flan-T5, and Llama deployed on GCP with containerization, load balancing, and autoscaling.
- **Resume-supported:** Observability included tokens/sec, latency, GPU utilization, error rates, including 429s, and capacity planning.

**Illustrative request path**

```text
Client request
   ↓
API / LLM gateway
   ↓
Prompt and context preparation
   ↓
Model-serving replica
   ↓
GPU inference runtime
   ↓
Post-processing / policy checks
   ↓
Response or downstream system
```

**Illustrative operating profile**

| Layer | Example operating concern |
|---|---|
| Gateway | Rate limits and request routing |
| Serving | Batch formation and queue time |
| GPU | Memory pressure and utilization |
| Application | Context size and output variability |
| Operations | 429s, retries, autoscaling lag |

**Visual placeholder**

`[Layered architecture diagram using the request path above]`

---

## 7. Anatomy of a Production LLM Request

**On-slide content**

**Illustrative request: “Summarize this email thread and create calendar actions.”**

| Stage | Illustrative latency | What influences it |
|---|---:|---|
| Request validation | 20 ms | Gateway and policy checks |
| Context assembly | 80 ms | Thread length, retrieval |
| LLM prefill | 450 ms | Prompt tokens, model, memory |
| Token generation | 1.2 s | Output length, runtime, batching |
| Post-processing | 100 ms | Structured output and filters |
| Downstream action | 250 ms | External API/service latency |
| **End-to-end** | **~2.1 s** | Full request path |

**Takeaway**

> The GPU is only one component of user-visible latency.

**Visual placeholder**

`[Waterfall chart of illustrative end-to-end latency]`

---

## 8. Inference Optimization Levers

**On-slide content**

- **Resume-supported:** Quantization: AWQ, GPTQ, GGUF.
- **Resume-supported:** FlashAttention and dynamic batching.
- **Resume-supported:** Serving/runtime tools: vLLM, TensorRT, ONNX, PyTorch, Triton.
- **Resume-supported:** Optimization work covered latency, throughput, memory footprint, and GPU utilization.

**Illustrative comparison**

| Configuration | Relative memory | Relative tokens/sec | Tradeoff to validate |
|---|---:|---:|---|
| FP16 baseline | 100% | 1.0× | Reference quality/performance |
| 8-bit quantized | 60% | 1.2× | Model-quality acceptance |
| 4-bit quantized | 35% | 1.4× | Accuracy and operational fit |
| Dynamic batching | Varies | 1.3–1.8× | TTFT under burst traffic |

**Footer**

> Illustrative values only. Actual results depend on model, prompt length, hardware, runtime, batch policy, and quality criteria.

**Visual placeholder**

`[Stack diagram: model precision → attention/runtime → batching/scheduling → GPU utilization]`

---

## 9. Resume-Reported Optimization Outcomes

**On-slide content**

- **Resume-supported:** Led 0-to-1 inference optimization for fine-tuned Qwen variants and Llama.
- **Resume-supported:** Reported **50%+ GPU cost reduction**.
- **Resume-supported:** Reported **45%+ latency improvement**.
- The resume does not isolate individual causes or provide a public benchmark configuration.

**Suggested visual**

```text
Resume-reported results
────────────────────────
50%+ reduction in GPU cost
45%+ improvement in latency
```

**Speaker note**

> I would present these as aggregate program outcomes. I would not imply that any one technique, model, or hardware choice created the full improvement.

---

## 10. Benchmarking Methodology for Production Inference

**On-slide content**

**Test dimensions**

| Dimension | Example values |
|---|---|
| Model | Fine-tuned Qwen, Flan-T5, Llama |
| Runtime | vLLM, TensorRT, ONNX, Triton |
| Hardware | A100, H100, G2-L4, G4 RTX PRO 6000 |
| Request shape | Short chat, long context, structured output |
| Load shape | Steady state, burst, sustained peak |
| Precision | FP16, 8-bit, 4-bit where applicable |

**Metrics**

- TTFT
- Tokens/sec
- P50 and P95 latency
- Memory footprint
- GPU utilization
- Error rate and 429s

**Illustrative test protocol**

1. Warm the service for **10 minutes**.
2. Run each load profile for **30 minutes**.
3. Capture P50/P95/P99 latency and errors.
4. Repeat across at least **three runs**.
5. Record model, runtime, precision, prompt-length, batch, and concurrency settings.

**Visual placeholder**

`[Benchmark matrix with rows = hardware/runtimes and columns = request/load profiles]`

---

## 11. Hardware Profiling: Comparisons Need Context

**On-slide content**

- **Resume-supported:** Hardware profiling included A100, H100, G2-L4, and G4 NVIDIA RTX PRO 6000.
- **Resume-supported:** Evaluation focused on latency, throughput, and memory footprint.
- A comparison remains meaningful only when the workload configuration is explicit.

**Illustrative benchmark record**

| Field | Example |
|---|---|
| Model | Fine-tuned 7B instruction model |
| Prompt / output | 1,500 input tokens / 300 output tokens |
| Runtime | vLLM version `[approved version]` |
| Precision | 4-bit AWQ |
| Batch policy | Continuous batching |
| Load | 32 concurrent clients |
| Metrics | TTFT, tokens/sec, P95 latency, GPU memory |

**Visual placeholder**

`[A blank “benchmark report card” template with the fields above]`

**Takeaway**

> Hardware results should travel with their workload assumptions.

---

## 12. Concurrency, Fleet Sizing, and Capacity Planning

**On-slide content**

- **Resume-supported:** GPU fleet sizing and client-side concurrency requirements were addressed for systems serving 1B+ daily requests.
- **Resume-supported:** GCP dashboards supported health monitoring and capacity planning.
- **Illustrative planning question:** At what request rate does P95 TTFT exceed the user-facing target?

**Illustrative load test**

| Concurrent clients | P95 TTFT | P95 end-to-end latency | GPU utilization | Error / 429 rate |
|---:|---:|---:|---:|---:|
| 8 | 450 ms | 2.2 s | 48% | 0.0% |
| 32 | 800 ms | 2.8 s | 71% | 0.1% |
| 64 | 1.6 s | 4.0 s | 86% | 0.6% |
| 96 | 3.4 s | 6.9 s | 92% | 2.4% |

**Interpretation**

> Illustrative example: throughput may continue rising after the user experience has crossed an acceptable latency threshold.

**Visual placeholder**

`[Line chart: concurrency on x-axis; P95 TTFT, GPU utilization, and error rate on separate axes]`

---

## 13. Agentic Workflows and Tool Use

**On-slide content**

- **Resume-supported:** Production agentic workflows enabled multi-step reasoning and tool use.
- **Resume-supported:** Experience includes LangChain, LangGraph, Google ADK, CrewAI, and MCP.
- **Resume-supported:** Systems included orchestration, memory/context frameworks, retrieval-reasoning hybrid architectures, and tool-use workflows.

**Illustrative workflow**

```text
User task
   ↓
Planner model call
   ↓
Retrieve account / document context
   ↓
Tool call: search, calendar, database, or API
   ↓
Verifier / policy check
   ↓
Final model response
```

**Illustrative workflow profile**

| Component | Illustrative calls | Typical bottleneck |
|---|---:|---|
| LLM inference | 3–6 | TTFT, decode speed, KV cache |
| Retrieval | 1–3 | Index/API latency |
| Tool invocation | 1–4 | External dependency latency |
| Verification | 1–2 | Extra model calls or rules |
| Total task time | 4–12 s | End-to-end orchestration |

**Visual placeholder**

`[Agent loop diagram with model, retrieval, tool, verifier, and memory nodes]`

---

## 14. Earlier Production Lessons: MLOps, RAG, and NLP

**On-slide content**

- **Resume-supported:** At Metrix, built enterprise MLOps platforms, RAG systems, GenAI gateways, monitoring, CI/CD, and cloud infrastructure for clients including Prudential.
- **Resume-supported:** Built distributed ML pipelines supporting **70,000+ models** for PepsiCo.
- **Resume-supported:** Delivered large-scale NLP and pricing models for Express Scripts/Cigna.
- **Resume-supported:** At Pearson, deployed neural-network essay scoring, plagiarism detection, and student-risk NLP systems.

**Cross-cutting lessons**

| Production concern | Why it remains relevant to LLM inference |
|---|---|
| Monitoring | Performance degrades without visibility |
| Deployment discipline | Model changes affect reliability and cost |
| Data and context quality | Weak inputs create weak outputs |
| Scale management | Capacity needs change with usage patterns |
| Evaluation | Speed alone does not establish usefulness |

**Visual placeholder**

`[Timeline with recurring layers underneath: data, platform, deployment, observability, evaluation]`

---

## 15. Discussion: Production-Informed Benchmarking for AMD

**On-slide content**

**Questions for the team**

1. Which request distributions should accompany model-level benchmarks?
2. Which latency target matters most for each workload: TTFT, token generation, or end-to-end completion?
3. How should tests represent dynamic batching, quantization, long context, and burst traffic?
4. Which reliability signals should become standard: queue depth, 429s, retries, or degraded-service behavior?
5. How can agentic workloads be represented without obscuring the hardware-level signal?

**Proposed next experiment**

| Phase | Deliverable |
|---|---|
| 1 | Define 3 representative request profiles |
| 2 | Run controlled model/runtime/hardware experiments |
| 3 | Add concurrency and burst-load tests |
| 4 | Compare model metrics with service outcomes |
| 5 | Build a reusable workload characterization suite |

**Visual placeholder**

`[Workshop board: representative workloads → benchmark design → measurement → insights]`

**Closing statement**

> The opportunity is to pair hardware excellence with workload representations that reflect how production AI services actually behave.
