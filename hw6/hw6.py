from keras import backend as K
from keras.models import load_model
from sklearn.cluster import KMeans
import csv
import numpy as np
import pandas as pd
import argparse


def test(path, label, output):
    out = []
    keys = ["ID", "Ans"]
    f = open(output, 'w')
    data = pd.read_csv(path).values
    for testcase in data:
        i1 = testcase[1]
        i2 = testcase[2]
        if (label[i1] == label[i2]):
            answer = '1'
        else:
            answer = '0'
        ans = {"ID":testcase[0], "Ans":answer}
        out.append(ans)

    o = csv.DictWriter(f, keys)
    o.writeheader()
    o.writerows(out)

def main(args):

    img = np.load(args.image).astype(np.float64)
    if (args.encoder=='CNN'):
        img = img.reshape(img.shape[0], 28,28,1).astype(np.float64)
    img = img/255
    model = load_model(args.model)
    img_reconstruct = model.predict(img)
    
    inp = model.input
    if (args.encoder == 'CNN'):
        output = model.layers[5].output
    else:
        output = model.layers[2].output
    func = K.function([inp], [output] )
    autoencoder_out = np.array(func([img])[0])
    reduced_data = autoencoder_out
        
    kmeans = KMeans(n_clusters=2).fit(reduced_data)
    label = kmeans.labels_
    unique, counts = np.unique(label, return_counts=True)
    c = dict(zip(unique, counts))
    test(args.test, label, args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image data", type=str, required=True)
    parser.add_argument("--test", help="testing data", type=str, required=True)
    parser.add_argument("--out", help="output data", type=str, required=True)
    parser.add_argument("--model", help="model file", type=str, required=True)
    parser.add_argument("--encoder", help="DNN or CNN", type=str, required=False, default="DNN")
    args = parser.parse_args()
    main(args)

