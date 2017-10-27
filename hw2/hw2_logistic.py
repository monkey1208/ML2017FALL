import csv
import sys
import numpy as np
import pandas as pd
import math
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def normalize(X_all, X_test):
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def predict(w, b, x):
    tmp = np.zeros(len(x))
    for i in range(len(x)):
        if sigmoid(np.inner(w, x[i]) + b) > 0.5:
            tmp[i] = int(1)
        else:
            tmp[i] = int(0)
    return tmp

def main():
    w = []
    b = 0
    with open(sys.argv[1], 'r') as f:
        fst = True
        for row in csv.reader(f):
            if fst:
                fst = False
                continue
            if row[0] != 'b':
                w.append(float(row[1]))
            else:
                b = float(row[1])
    w = np.array(w)
    data = pd.read_csv(sys.argv[2])
    data_test = pd.read_csv(sys.argv[3])
    X_train = data.values.astype(np.float)
    X_test = data_test.values.astype(np.float)
    X_train, X_test = normalize(X_train, X_test)
    result = predict(w, b, X_test)
    
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
