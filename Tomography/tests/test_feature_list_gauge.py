import pytest
from src.Tomography_qutrit import*
from numpy import allclose

protocol_QWP = [Gl_4(0), Gl_4(pi / 8), Gl_4(3 * pi / 8), Gl_4(5 * pi / 8), Gl_4(7 * pi / 8)]

@pytest.mark.parametrize(
        "protocol, rank, sigma1, sigma2, sigma3, phh, khh",
        [
        (protocol_QWP, 2, 10.7, 100.1, 69.15, tomography_pol_qutrit(protocol_QWP).hh,
         array([72.7, 5.35,	1.15,
                44.6, 58.1, 20.95,
                34.55, 94.55, 12.95,
                39.9, 78.3, 10.7,
                37.35, 73.65, 22.3])),
        ((protocol_QWP, 1, 10.7, 100.1, 69.15, tomography_pol_qutrit(protocol_QWP).hh,
         array([72.7, 5.35,	1.15,
                44.6, 58.1, 20.95,
                34.55, 94.55, 12.95,
                39.9, 78.3, 10.7,
                37.35, 73.65, 22.3]))),
        ]  
)
def test_diff_sigma(protocol, rank, sigma1, sigma2, sigma3, phh, khh):
    
    Start_protocol = tomography_pol_qutrit(protocol)

    Start_protocol.experiment(khh, phh, sigma1, sigma2, sigma3, rank, visible=False)
    matrix_1 = Start_protocol.matrix_finish

    length = len(protocol)
    sigma1 = full(length, sigma1)
    sigma2 = full(length, sigma2)
    sigma3 = full(length, sigma3)
    print(type(sigma2))
    print(type(khh[::3]))
    print(khh[::3] / sigma2)
    Start_protocol.experiment(khh, phh, sigma1, sigma2, sigma3, rank, visible=False)
    matrix_2 = Start_protocol.matrix_finish

    assert allclose(matrix_1, matrix_2, atol=1e-5)