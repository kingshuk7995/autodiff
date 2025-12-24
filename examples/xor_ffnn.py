# example for feed forward neural network on XOR

import numpy as np
from autodiff import Tensor

# xor data
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

Y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

np.random.seed(42)

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = Tensor(
            np.random.randn(in_dim, out_dim) * 0.1,
            requires_grad=True
        )
        self.b = Tensor(
            np.zeros((1, out_dim)),
            requires_grad=True
        )

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

fc1 = Linear(2, 8)
fc2 = Linear(8, 1)

# def softmax(self) -> Tensor:
#     shifted = self.data - np.max(self.data)
#     exps = np.exp(shifted)
#     out_data = exps / exps.sum()
#     out = Tensor(out_data,
#                 parents=(self,),
#                 op_name="Softmax",
#                 requires_grad=self.requires_grad,
#             )
#     def _backward_fn():
#         if self.requires_grad:
#             s = out.data
#             grad_sum = np.sum(out.grad * s)
#             self.grad += (out.grad - grad_sum) * s
#     out._backward_fn = _backward_fn
#     return out

def model(x: Tensor) -> Tensor:
    h = fc1(x).relu()
    y = fc2(h)
    return y.sigmoid()

# error function
def mse(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred + Tensor(-target.data, requires_grad=False)
    return (diff * diff).mean()

# train loop
lr = 0.1
params = fc1.parameters() + fc2.parameters()

for epoch in range(5000):
    # zero grads
    for p in params:
        p.zero_grad()

    x = Tensor(X)
    y_true = Tensor(Y, requires_grad=False)

    y_pred = model(x)
    loss = mse(y_pred, y_true)

    loss.backward()

    # SGD update
    for p in params:
        assert p.grad is not None
        p.data -= lr * p.grad

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d} | loss = {loss.data:.6f}")

# final test
with np.printoptions(precision=3, suppress=True):
    preds = model(Tensor(X)).data
    print("\nPredictions:")
    print(preds)
