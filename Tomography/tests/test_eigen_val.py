import pytest
from purification_state import Purification 
from Qutrit import Gl_4
import numpy as np







@pytest.mark.parametrize(
    "rank, dim, protocol",
    [   
        # протокол из статьи богданова
        (1, 3, None),
        (2, 3, None),
        (3, 3, None),

        # протокол с пластинкой λ/4
        (1, 3, [Gl_4(0), Gl_4(np.pi/8), Gl_4(3 * np.pi/8), Gl_4(5 * np.pi/8), Gl_4(7 * np.pi/8)]),
        (2, 3, [Gl_4(0), Gl_4(np.pi/8), Gl_4(3 * np.pi/8), Gl_4(5 * np.pi/8), Gl_4(7 * np.pi/8)]),
        (3, 3, [Gl_4(0), Gl_4(np.pi/8), Gl_4(3 * np.pi/8), Gl_4(5 * np.pi/8), Gl_4(7 * np.pi/8)])
      ]
)
def test_result(rank, dim, protocol):

    pure_class = Purification(rank = rank, dimension = dim, protocol = protocol)
    val = (np.real(np.round(pure_class.value_matrix_information(), 14)))
    val = list(np.sign(val))
    numb_zero = val.count(0)

    assert rank**2 == numb_zero

