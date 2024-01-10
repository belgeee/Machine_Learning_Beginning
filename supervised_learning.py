# shugaman algebr
import pandas as pd
# uguggultei ajillah san
import numpy as np
# grapiktai ajillah san
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# Assuming you have a CSV file named 'dataset.csv' in the current directory
# data_path = "hello/sad"
# data = pd.read_csv('dataset.csv')
# X = data.iloc[:, 0]
# Y = data.iloc[:, 1]

X = [0, 1, 12, 32, 12, 3]
Y = [0, 3, 32, 42, 12, 4]


# durs baiguulj bn
plt.scatter(X, Y)
# plt.show()
# zagwaraa uusgej bn
w1 = 0
w0 = 0
Y_hat = w1 * np.array(X) + w0

n = float(len(X))
alpha = 0.001
epoch = 1000


# gradient to create linear algebr function 
for i in range(epoch):
    print(i, "th epoch")
    Y_hat = w1 * np.array(X) + w0
    L_w1 = (-2/n) * sum(np.array(X) * (np.array(Y) - Y_hat))
    L_w0 = (-2/n) * sum(np.array(Y) - Y_hat)
    w1 = w1 - alpha * L_w1
    w0 = w0 - alpha * L_w0

print(w1, w0)
Y_hat = w1 * np.array(X) + w0

plt.plot([min(X), max(X)], [min(Y_hat), max(Y_hat)], color="green")
plt.show()

print(dir(plt))
print(dir(pd))
print(dir(np))

