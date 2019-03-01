from code import mean_squared_error
from sklearn import metrics
from .test_utils import make_fake_data

def test_mean_squared_error():
    y_pred, y_true = make_fake_data()

    _est = mean_squared_error(y_pred, y_true)
    _gt = metrics.mean_squared_error(y_true, y_pred)

    assert (_est == _gt)