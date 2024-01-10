import numpy as np
import pandas as pd

y_hat=np.array([0.00, 0.166, 0.1321])
y=np.array([0.01, 0.266, 0.4321]);
def rmse(prediction, target):
    print("error functionn started")
    differences=prediction-target
    differences_squarted=differences**2
    mean_differences_squarted=differences_squarted.mean()
    rmse_val=np.sqrt(mean_differences_squarted)
    return rmse_val


def rms(prediction, target):
    differences=prediction-target
    avg_differences=differences.mean();
    return avg_differences


test1_val=rms(y_hat, y);

test_val=rmse(y_hat, y)
print("bataaa ",test_val)
print(dir(np))
print("dorjoo", test1_val);
print(dir(pd))
