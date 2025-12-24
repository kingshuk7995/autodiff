from torchvision.datasets import MNIST
import numpy as np
from autodiff import Tensor


# Load MNIST (torchvision only for data)
train_ds = MNIST(root="./data", train=True, download=True)
test_ds  = MNIST(root="./data", train=False, download=True)

X_train = train_ds.data.numpy().astype(np.float32) / 255.0
y_train = train_ds.targets.numpy().astype(np.int64)

X_test  = test_ds.data.numpy().astype(np.float32) / 255.0
y_test  = test_ds.targets.numpy().astype(np.int64)

# flatten
X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test), -1)

np.random.seed(42)

# Layers
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

# model
fc1 = Linear(784, 256)
fc2 = Linear(256, 256)
fc3 = Linear(256, 10)

def model(x: Tensor) -> Tensor:
    h1 = fc1(x).relu()
    h2 = fc2(h1).relu()
    logits = fc3(h2)
    return logits

# utils
def batch_iter(X, y, batch_size=128, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]

# softmax + cross entropy
def softmax_cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    x = logits.data
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    B = x.shape[0]
    loss_val = -np.log(probs[np.arange(B), y]).mean()
    loss = Tensor(loss_val, parents=(logits,), op_name="SoftmaxCE")

    def _backward_fn():
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(B), y] -= 1.0
            grad /= B
            logits.grad += grad

    loss._backward_fn = _backward_fn
    return loss

# training
params = (
    fc1.parameters() +
    fc2.parameters() +
    fc3.parameters()
)

lr = 0.1
epochs = 10
batch_size = 128

for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    seen = 0

    for Xb, yb in batch_iter(X_train, y_train, batch_size=batch_size):
        # zero gradients
        for p in params:
            p.zero_grad()

        x = Tensor(Xb, requires_grad=False)
        logits = model(x)

        loss = softmax_cross_entropy(logits, yb)
        loss.backward()

        # SGD
        for p in params:
            p.data -= lr * p.unwrap_grad()

        total_loss += loss.data * len(Xb)
        preds = np.argmax(logits.data, axis=1)
        correct += (preds == yb).sum()
        seen += len(Xb)

    print(
        f"epoch {epoch+1:02d} | "
        f"loss = {total_loss / seen:.4f} | "
        f"acc = {correct / seen:.4f}"
    )

# evaluation
correct = 0
seen = 0

for Xb, yb in batch_iter(X_test, y_test, batch_size=256, shuffle=False):
    x = Tensor(Xb, requires_grad=False)
    logits = model(x)
    preds = np.argmax(logits.data, axis=1)
    correct += (preds == yb).sum()
    seen += len(Xb)

print(f"\nTest accuracy: {correct / seen:.4f}")
