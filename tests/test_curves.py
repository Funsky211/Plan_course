import numpy as np
import pytest

from trailnutri.curves import gaussian_window, ramp, sigmoid

X = np.linspace(-1, 1, 201)


# --- sigmoid -------------------------------------------------------------------------------

def test_sigmoid_centre_et_bornes():
    assert sigmoid(0.08, centre=0.08, raideur=50) == pytest.approx(0.5)
    assert sigmoid(-10, 0, 50) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(10, 0, 50) == pytest.approx(1.0)
    y = sigmoid(X, 0, 5)
    assert ((y >= 0) & (y <= 1)).all()


def test_sigmoid_monotone_et_symetrique():
    y = sigmoid(X, 0.1, 8)
    assert (np.diff(y) > 0).all()
    # Symétrie autour du centre : s(c + d) + s(c - d) = 1
    assert sigmoid(0.1 + 0.03, 0.1, 30) + sigmoid(0.1 - 0.03, 0.1, 30) == pytest.approx(1.0)


def test_sigmoid_raideur():
    # Plus raide = plus proche de 1 à la même distance au-dessus du centre
    assert sigmoid(0.1, 0, 50) > sigmoid(0.1, 0, 10)


def test_sigmoid_pas_d_overflow():
    with np.errstate(over="raise"):
        assert sigmoid(1e6, 0, 1e3) == pytest.approx(1.0)


# --- ramp ----------------------------------------------------------------------------------

def test_ramp_bornes_et_milieu():
    assert ramp(0, 1, 3) == 0
    assert ramp(1, 1, 3) == 0
    assert ramp(2, 1, 3) == pytest.approx(0.5)
    assert ramp(3, 1, 3) == 1
    assert ramp(10, 1, 3) == 1


def test_ramp_monotone():
    assert (np.diff(ramp(X, -0.5, 0.5)) >= 0).all()


# --- gaussian_window -----------------------------------------------------------------------

def test_gaussienne_centre_et_largeur():
    assert gaussian_window(3.25, centre=3.25, largeur=0.75) == pytest.approx(1.0)
    assert gaussian_window(3.25 + 0.75, 3.25, 0.75) == pytest.approx(np.exp(-0.5))
    assert gaussian_window(100, 3.25, 0.75) == pytest.approx(0.0, abs=1e-12)


def test_gaussienne_symetrique_et_unimodale():
    d = np.linspace(0, 3, 50)
    assert gaussian_window(1 + d, 1, 0.5) == pytest.approx(gaussian_window(1 - d, 1, 0.5))
    y = gaussian_window(X, 0, 0.3)
    centre = len(X) // 2
    assert (np.diff(y[: centre + 1]) > 0).all() and (np.diff(y[centre:]) < 0).all()


# --- paramètres invalides ------------------------------------------------------------------

@pytest.mark.parametrize(
    "appel",
    [
        lambda: sigmoid(0, 0, 0),
        lambda: sigmoid(0, 0, -1),
        lambda: ramp(0, 1, 1),
        lambda: ramp(0, 2, 1),
        lambda: gaussian_window(0, 0, 0),
    ],
)
def test_parametres_invalides(appel):
    with pytest.raises(ValueError):
        appel()
