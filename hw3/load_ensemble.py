from keras.models import load_model
import h5py
import numpy as np
import pandas as pd
import sys, csv

def readFile(path):
    data = pd.read_csv(path).values
    X_test = []
    for i in data[:, 1]:
        X_test.append(np.fromstring(i, dtype=float, sep=' '))
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    X_test = X_test/255
    return X_test

def output(model1, model2, model3, model4, model5, model6, outputpath, X_test):
    model1 = load_model(model1)
    model2 = load_model(model2)
    model3 = load_model(model3)
    model4 = load_model(model4)
    model5 = load_model(model5)
    model6 = load_model(model6)
    result1 = model1.predict(X_test)
    result2 = model2.predict(X_test)
    result3 = model3.predict(X_test)
    result4 = model4.predict(X_test)
    result5 = model5.predict(X_test)
    result6 = model6.predict(X_test)
    label1 = []
    label2 = []
    label3 = []
    label4 = []
    label5 = []
    label6 = []
    label = []
    out = []
    
    for i in range(len(result1)):
        label1.append(int(np.where(result1[i].max()==result1[i])[0][0]))
        label2.append(int(np.where(result2[i].max()==result2[i])[0][0]))
        label3.append(int(np.where(result3[i].max()==result3[i])[0][0]))
        label4.append(int(np.where(result4[i].max()==result4[i])[0][0]))
        label5.append(int(np.where(result5[i].max()==result5[i])[0][0]))
        label6.append(int(np.where(result6[i].max()==result6[i])[0][0]))
    for i in range(len(label1)):
        tmp = [0,0,0,0,0,0,0]
        tmp[label1[i]] += 1
        tmp[label2[i]] += 1
        tmp[label3[i]] += 1
        tmp[label4[i]] += 1
        tmp[label5[i]] += 1
        tmp[label6[i]] += 1
        t_max = 0
        t_index = -1
        for j in range(len(tmp)):
            if tmp[j] > t_max:
                t_max = tmp[j]
                t_index = j
            elif tmp[j] == t_max:
                if tmp[j] == 3:
                    t_index = label1[i]
                elif tmp[j] == 2:
                    if label1[i] == t_index:
                        t_index = label1[i]
                    elif label2[i] == t_index:
                        t_index = label2[i]
                    else:
                        t_index = label3[i]
        if t_max == 1:
            t_index = label1[i]
        label.append(t_index)
    for i in range(len(label)):
        ans = {"id":i, "label":label[i]}
        out.append(ans)
    with open(outputpath, 'w') as f:
        o = csv.DictWriter(f, out[0].keys())
        o.writeheader()
        o.writerows(out)

X_test = readFile(sys.argv[1])
output(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], X_test)
