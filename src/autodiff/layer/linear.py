import numpy as np
from ..tensor import Tensor

class Linear:
    def __init__(self, in_dim, out_dim, rng:np.random.Generator):
        if rng is None:
            # enforcing best practices
            raise AttributeError(f'rng must be non null')
        self.W = Tensor(
            rng.random((in_dim, out_dim)) * 0.1,
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
