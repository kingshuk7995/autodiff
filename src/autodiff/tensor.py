from __future__ import annotations
from typing import Optional, Callable
import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(
        self,
        data: list | np.ndarray,
        parents: tuple[Tensor, ...] = (),
        op_name: str = "",
        requires_grad: bool = True,
    ):
        self.data = np.array(data, dtype=float)
        self._parents = parents
        self._op_name = op_name
        self.requires_grad = requires_grad

        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward_fn: Callable[[], None] = lambda: None

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad: Optional[np.ndarray] = None):
        topo = []
        visited = set()

        def build_topo(v: Tensor):
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = grad

        for node in reversed(topo):
            node._backward_fn()

    # ---------------- ops ----------------

    def __add__(self, other: Tensor) -> Tensor:
        out = Tensor(
            self.data + other.data,
            parents=(self, other),
            op_name="+",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward_fn():
            assert out.grad is not None
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward_fn = _backward_fn
        return out

    def __mul__(self, other: Tensor) -> Tensor:
        out = Tensor(
            self.data * other.data,
            parents=(self, other),
            op_name="*",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward_fn():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward_fn = _backward_fn
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(
            self.data @ other.data,
            parents=(self, other),
            op_name="@",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward_fn = _backward_fn
        return out

    def relu(self) -> Tensor:
        out = Tensor(
            np.maximum(0, self.data),
            parents=(self,),
            op_name="ReLU",
            requires_grad=self.requires_grad,
        )

        def _backward_fn():
            if self.requires_grad:
                assert out.grad is not None and self.grad is not None
                self.grad += (out.data > 0) * out.grad

        out._backward_fn = _backward_fn
        return out

    def sum(self) -> Tensor:
        out = Tensor(
            np.array(self.data.sum()),
            parents=(self,),
            op_name="Sum",
            requires_grad=self.requires_grad,
        )

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)

        out._backward_fn = _backward_fn
        return out

    def mean(self) -> Tensor:
        out = Tensor(
            np.array(self.data.mean()),
            parents=(self,),
            op_name="Mean",
            requires_grad=self.requires_grad,
        )

        def _backward_fn():
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad * np.ones_like(self.data) / self.data.size

        out._backward_fn = _backward_fn
        return out
    
    def sigmoid(self) -> Tensor:
        out_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(out_data, parents=(self,), op_name="Sigmoid")

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * out.data * (1 - out.data)

        out._backward_fn = _backward_fn
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def unwrap_grad(self) -> np.ndarray:
        assert self.grad is not None
        return self.grad
