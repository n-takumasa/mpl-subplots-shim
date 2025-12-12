import pytest
from matplotlib.axes import Axes
from mpl_subplots_shim import Grid1D, Grid2D, subplots
from typing_extensions import Any, assert_type


@pytest.fixture
def unknown_int():
    return 3

def test_subplots(unknown_int: int):
    assert_type(subplots(squeeze=True)[-1], Axes)
    assert_type(subplots(1, 1, squeeze=True)[-1], Axes)
    assert_type(subplots(2, 1, squeeze=True)[-1], Grid1D[Axes])
    assert_type(subplots(1, 2, squeeze=True)[-1], Grid1D[Axes])
    assert_type(subplots(ncols=2, squeeze=True)[-1], Grid1D[Axes])
    assert_type(subplots(2, 2, squeeze=True)[-1], Grid2D[Axes])

    assert_type(subplots(unknown_int, 1, squeeze=True)[-1], Any)
    assert_type(subplots(1, unknown_int, squeeze=True)[-1], Any)
    assert_type(subplots(unknown_int, unknown_int, squeeze=True)[-1], Any)

    assert_type(subplots(unknown_int, 1, squeeze=False)[-1], Grid2D[Axes])
    assert_type(subplots(1, unknown_int, squeeze=False)[-1], Grid2D[Axes])
    assert_type(subplots(unknown_int, unknown_int, squeeze=False)[-1], Grid2D[Axes])
