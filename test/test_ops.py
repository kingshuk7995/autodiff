# tests/test_autodiff.py
import autodiff as ad
import numpy as np

def test_add_forward():
    a = ad.Tensor(np.array([1.0, 2.0]))
    b = ad.Tensor(np.array([3.0, 4.0]))

    c = a + b

    assert np.allclose(c.data, np.array([4.0, 6.0]))

def test_mul_forward():
    a = ad.Tensor(np.array([1.0, 2.0]))
    b = ad.Tensor(np.array([3.0, 4.0]))

    c = a * b
    correct =  a.data * b.data

    assert np.allclose(c.data, correct)

def test_matmul_forward():
    a = ad.Tensor(np.array([1.0, 2.0]))
    b = ad.Tensor(np.array([3.0, 4.0]).T)

    c = a @ b
    correct =  a.data @ b.data

    assert np.allclose(c.data, correct)

def test_relu_forward():
    a = ad.Tensor(np.array([1.0, -1.0]))

    b = a.relu()

    assert np.allclose(b.data, np.array([1.0, 0.0]))
