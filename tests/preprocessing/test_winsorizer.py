import numpy as np
import pytest
from ps3.preprocessing import Winsorizer

@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    wins = Winsorizer(lower_quantile, upper_quantile)
    wins.fit(X)

    # quantiles computed correctly
    assert wins.lower_quantile_ == np.quantile(X, lower_quantile)
    assert wins.upper_quantile_ == np.quantile(X, upper_quantile)

    # clipping works
    X_t = wins.transform(X)
    assert X_t.min() >= wins.lower_quantile_ # The minimum value of the quantile is greater than or equal to the lower quantile
    assert X_t.max() <= wins.upper_quantile_

    assert X_t.shape == X.shape # Checks the shape is unchanged

    # special case: same quantile
    if lower_quantile == upper_quantile:
        assert np.allclose(X_t, wins.lower_quantile_)
