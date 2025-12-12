
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from mpl_subplots_shim._grid import Grid1D, Grid2D
from typing_extensions import assert_type

nrows = 2
ncols = 3

@pytest.fixture
def ax() -> Grid2D[Axes]:
    plt.switch_backend("agg")
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    return cast(Grid2D[Axes], ax)


def test_grid2d(ax: Grid2D[Axes]):
    _ = ax[()]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[0]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0,]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0, 0]
    assert_type(_, Axes)
    assert isinstance(_, Axes)

    _ = ax[:]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[:,]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[:, :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[0, :]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[:, 0]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows,)

    _ = ax[...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[...,]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[..., :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[:, ...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[..., 0]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows,)
    _ = ax[0, ...]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)

    _ = ax[[True] * nrows]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)

    _ = ax[[]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (0, ncols)
    _ = ax[[0]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[(0,),]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[[0],]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)

    _ = ax[(0,), (0,)]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
    _ = ax[(0, 0), (0, 0)]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (2,)
    _ = ax[[0], [0]]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
    _ = ax[[0, 0], [0, 0]]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (2,)

    _ = ax[(0,), :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[(0,), ...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[[0], :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[[0], ...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)

    _ = ax[:, (0,)]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)
    _ = ax[..., (0,)]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)
    _ = ax[:, [0]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)
    _ = ax[..., [0]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)

    _ = ax[np.ones(nrows, dtype=np.bool)]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[np.array([0], dtype=np.intp)]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)

    _ = ax[np.full((nrows, ncols), True, dtype=np.bool)]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows * ncols,)
    _ = ax[[[True] * ncols] * nrows]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows * ncols,)

def test_grid1d(ax: Grid2D[Axes]):
    _ = ax[0][0]
    assert_type(_, Axes)
    assert isinstance(_, Axes)
    _ = ax[0][:]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][...]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][()]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][0,]
    assert_type(_, Axes)
    assert isinstance(_, Axes)

    _ = ax[0][:,]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][[],]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (0,)
    _ = ax[0][(),]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (0,)
    _ = ax[0][[0]]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
    _ = ax[0][(0,),]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
