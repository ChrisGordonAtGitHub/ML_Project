import numpy as np

def affine_forward(x, w, b):
    out = None
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    out = np.dot(x_row, w) + b
    cache = (x, w, b)
    return out, cache
  
def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    x_row = x.reshape(x.shape[0], -1)
    dw = np.dot()
    
X = np.array([[2,1],  
            [-1,1],  
            [-1,-1],  
            [1,-1]])      # 用于训练的坐标，对应的是I、II、III、IV象限
t = np.array([0,1,2,3])   # 标签，对应的是I、II、III、IV象限
np.random.seed(1)         # 有这行语句，你们生成的随机数就和我一样了

# 一些初始化参数  
input_dim = X.shape[1]     # 输入参数的维度，此处为2，即每个坐标用两个数表示
num_classes = t.shape[0]   # 输出参数的维度，此处为4，即最终分为四个象限
hidden_dim = 50            # 隐藏层维度，为可调参数
reg = 0.001                # 正则化强度，为可调参数
epsilon = 0.001            # 梯度下降的学习率，为可调参数
# 初始化W1，W2，b1，b2
W1 = np.random.randn(input_dim, hidden_dim)     # (2,50)
W2 = np.random.randn(hidden_dim, num_classes)   # (50,4)
b1 = np.zeros((1, hidden_dim))                  # (1,50)
b2 = np.zeros((1, num_classes))                 # (1,4)

for j in range(10000):
 # ①前向传播
    H, fc_cache = affine_forward(X,W1,b1)
    H = np.maximum(0, H)
    relu_cache = H
    Y, cachey = affine_forward(H,W2,b2)
 # ②Softmax层计算
    probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))
    probs /= np.sum(probs)
    
