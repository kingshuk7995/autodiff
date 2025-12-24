from autodiff import Tensor
from autodiff.layer import Linear
import numpy as np

def test_import():
    # testing if importable
    fc1 = Linear(1, 1, np.random.default_rng())
    assert np.allclose(fc1(Tensor([0])).data, [0])
