import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_data = pd.read_csv('data.csv', header=None)
sonar_data.head()
# sonar_data.shape()
sonar_data.describe() 