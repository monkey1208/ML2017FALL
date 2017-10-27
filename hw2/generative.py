import pandas as pd
import numpy as np
import sys, csv, math

def sigmoid(x):
    if x>=0:
        z = math.exp(-x)
        return 1/(1+z)
    else:
        z = math.exp(x)
        return z/(1+z)
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

def main():
    data = pd.read_csv(sys.argv[1])
    data2 = pd.read_csv(sys.argv[2])
    data_test = pd.read_csv(sys.argv[3])
    X_train = data.values.astype(np.float)
    Y_train = data2.values.ravel()
    X_test = data_test.values.astype(np.float)
    X_train, X_test = normalize(X_train, X_test)
    c0_index, = np.where( Y_train==0 )
    c1_index, = np.where( Y_train==1 )
    c0 = X_train[c0_index, :]
    c1 = X_train[c1_index, :]
    mu0 = c0.mean(axis=0)
    mu1 = c1.mean(axis=0)
    sig0 = np.cov(c0, rowvar=0)
    sig1 = np.cov(c1, rowvar=0)
    sig = (sig0*len(c0) + sig1*len(c1))/(len(c1)+len(c0))
    sig_inv = np.linalg.pinv(sig)
    w0 = (np.dot((mu0-mu1), sig_inv)).transpose()
    w1 = (np.dot((mu1-mu0), sig_inv)).transpose()
    b0 = (-0.5)*(np.dot(np.dot(mu0, sig_inv), mu0))+(0.5)*(np.dot(np.dot(mu1, sig_inv), mu1))+np.log(len(c0)/len(c1))
    b1 = (-0.5)*(np.dot(np.dot(mu1, sig_inv), mu1))+(0.5)*(np.dot(np.dot(mu0, sig_inv), mu0))+np.log(len(c1)/len(c0))
    print(w0)
    print(w1)
    print(b0)
    print(b1)
    output = []
    for i in range(len(X_test)):
        p0 = sigmoid(np.inner(w0, X_test[i])+b0)
        p1 = sigmoid(np.inner(w1, X_test[i])+b1)
        ans = {"id":0, "label":0}
        if p0>=p1:
            ans = {"id":i+1, "label":0}
        else:
            ans = {"id":i+1, "label":1}
        output.append(ans)
    with open(sys.argv[4], 'w') as f:
        o = csv.DictWriter(f, output[0].keys())
        o.writeheader()
        o.writerows(output)
    
if __name__ == "__main__":
    main()
