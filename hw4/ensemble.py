from keras.models import load_model
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
import h5py
import numpy as np
import pandas as pd
import sys, csv

def readFile(path, modelw2v):
    sequence = []
    e_index = 0
    n = 0
    with open(path, 'r') as f:
        tmp = True
        for line in f:
            if tmp == True:
                tmp = False
                continue
            line = (line.strip('\n').split(',',1))[1]
            seq = line.split(' ')
            sequence.append([])
            for word in seq:
                word = word.encode('utf-8').decode('utf-8')
                try:
                    w2v = modelw2v.wv[word]
                    sequence[e_index].append(w2v)
                except:
                    continue
            e_index += 1
        sequence = np.array(sequence)
        sequence = pad_sequences(sequence, maxlen=40, dtype='float64', padding='post', truncating='post', value=0)
    return sequence

def output5(outputpath):
    test_path = sys.argv[1]
    
    modelw2v = word2vec.Word2Vec.load("models/word_embedding250.model.bin")
    test = readFile(test_path, modelw2v)
    model1 = load_model("models/1.h5")
    result1 = model1.predict(test)

    test = readFile(test_path, modelw2v)
    model2 = load_model("models/2.h5")
    result2 = model2.predict(test)

    modelw2v = word2vec.Word2Vec.load("models/word_embedding100.model.bin")
    test = readFile(test_path, modelw2v)
    model3 = load_model("models/5.h5")
    result3 = model3.predict(test)

    test = readFile(test_path, modelw2v)
    model4 = load_model("models/6.h5")
    result4 = model4.predict(test)

    modelw2v = word2vec.Word2Vec.load("models/word_embedding50.model.bin")
    test = readFile(test_path, modelw2v)
    model5 = load_model("models/4.h5")
    result5 = model5.predict(test)

    length = len(result1)
    label1 = np.zeros(length)
    label2 = np.zeros(length)
    label3 = np.zeros(length)
    label4 = np.zeros(length)
    label5 = np.zeros(length)
    i1 = np.where(result1>0.5)
    for i in i1:
        label1[i] = 1
    i2 = np.where(result2>0.5)
    for i in i2:
        label2[i] = 1
    i3 = np.where(result3>0.5)
    for i in i3:
        label3[i] = 1
    i4 = np.where(result4>0.5)
    for i in i4:
        label4[i] = 1
    i5 = np.where(result5>0.5)
    for i in i5:
        label5[i] = 1
    out = []

    for i in range(len(label1)):
        tmp = [0,0]
        tmp[int(label1[i])] += 1
        tmp[int(label2[i])] += 1
        tmp[int(label3[i])] += 1
        tmp[int(label4[i])] += 1
        tmp[int(label5[i])] += 1
        label = 0
        if tmp[0] <= 2:
            label = 1
        ans = {"id":i, "label":label}
        out.append(ans)
    with open(outputpath, 'w') as f:
        o = csv.DictWriter(f, out[0].keys())
        o.writeheader()
        o.writerows(out)
output5(sys.argv[2])
