import pandas as pd
import numpy as np
import sys
import math
import csv
import signal
from sklearn.ensemble import GradientBoostingClassifier

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma
    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def main():
    data = pd.read_csv(sys.argv[1])
    data2 = pd.read_csv(sys.argv[2])
    arr = np.array(data.values)
    arr = arr.astype(np.float)
    data_test = pd.read_csv(sys.argv[3])
    X_test = np.array(data_test.values.astype(np.float))
    X_train, X_test = normalize(arr, X_test)
    label = np.array(data2.values)
    model = GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=1000)
    print("Start Training")
    model.fit(X_train, label.ravel())
    print("Training Finish")
    va = model.predict(X_train[10000:20000])
    #print("acc = ", float((va==label.ravel()[10000:20000]).sum())/10000)
    test(model, X_test)

def test(model, X_test):
    result = model.predict(X_test)
    output = []
    for i in range(len(result)):
        ans = {"id":i+1, "label":int(result[i])}
        output.append(ans)
    with open(sys.argv[4], 'w') as f:
        keys = output[0].keys()
        o = csv.DictWriter(f, keys)
        o.writeheader()
        o.writerows(output)

if __name__ == "__main__":
    main()
