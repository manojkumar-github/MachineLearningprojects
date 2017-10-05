def forward(x, W1, W2, W3, training=False):
    z1 = np.dot(x, W1)
    y1 = np.tanh(z1)
 
    z2 = np.dot(y1, W2)
    y2 = np.tanh(z2)
    # Dropout in layer 2
    if training:
        m2 = np.random.binomial(1, 0.5, size=z2.shape)
    else:
        m2 = 0.5
    y2 *= m2
 
    z3 = np.dot(y2, W3)
    y3 = z3 # linear output
 
    return y1, y2, y3, m2
 
def backward(x, y1, y2, y3, m2, t, W1, W2, W3):
    dC_dz3 = dC(y3, t)
    dC_dW3 = np.dot(y2.T, dC_dz3)
    dC_dy2 = np.dot(dC_dz3, W3.T)
 
    dC_dz2 = dC_dy2 * dtanh(y2) * m2
    dC_dW2 = np.dot(y1.T, dC_dz2)
    dC_dy1 = np.dot(dC_dz2, W2.T)
 
    dC_dz1 = dC_dy1 * dtanh(y1)
    dC_dW1 = np.dot(x.T, dC_dz1)
 
    return dC_dW1, dC_dW2, dC_dW3
