import autodiff as ad
import numpy as np

def test_simple_backward():
    x = ad.Tensor([2.0])
    y = ad.Tensor([3.0])

    f = x * y + y
    f.backward()
    assert np.isclose(x.unwrap_grad(), [3.0])
    assert np.isclose(y.unwrap_grad(), [2.0 + 1.0])

def test_dag_backward1():
    x = ad.Tensor([2.0])
    y1 = x * x
    y2 = x * x
    z = y1 + y2
    z.sum().backward()
    assert np.allclose(x.unwrap_grad(), [8.0])

def test_dag_backward2():
    x = ad.Tensor([-1.0])
    y = x.relu()
    z = y + y
    z.sum().backward()


def test_dag_nonlinear_reuse():
    x = ad.Tensor([2.0])

    y = x * x          # y = 4
    u = y.relu()       # u = 4
    z = y * u          # z = 16

    z.sum().backward()

    # Correct gradient:
    # z = y * u
    # dz/dx = dy/dx * u + y * du/dx
    #        = (2x)*4 + 4*(2x) = 16 + 16 = 32
    assert np.allclose(x.unwrap_grad(), [32.0])

def test_relu_gate_with_shared_parent():
    x = ad.Tensor([-1.0, 2.0])

    y = x * x              # [1, 4]
    u = y.relu()           # [1, 4]
    z = (y + u).sum()

    z.backward()

    # Correct gradient:
    # y = x^2
    # z = y + relu(y)
    # dz/dx = 2x + 2x * 1[y>0]
    # x = [-1, 2]
    # dz/dx = [-2 + -2, 4 + 4] = [-4, 8]

    assert np.allclose(x.unwrap_grad(), [-4.0, 8.0])

def test_softmax_reuse_fails():
    x = ad.Tensor([1.0, 2.0, 3.0])

    y = x.softmax()
    z = y + y
    z.sum().backward()

    # Correct gradient is ZERO
    # because sum(softmax(x)) = 1 → constant
    assert np.allclose(x.unwrap_grad(), [0.0, 0.0, 0.0])

def test_diamond():
    a = ad.Tensor([2.0])
    b = a * a          # b = a^2
    b.backward()       # d(b)/da = 2a = 4.0

    assert np.allclose(a.unwrap_grad(), [4.0])

    x = ad.Tensor([3.0])
    a = x + x          # a = 2x = 6.0
    b = a * a          # b = a^2 = (2x)^2 = 4x^2
    b.backward()       # db/dx = 8x = 24.0

    assert np.allclose(x.unwrap_grad(), [24.0])
