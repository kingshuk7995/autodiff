import numpy as np

class Adam:
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # first moment => momentum
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # second moment => RMS
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
