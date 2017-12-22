from keras.models import load_model
import pandas as pd
import numpy as np
import argparse

import h5py
import numpy as np
import pandas as pd
import sys, csv

def main(args):
    testing = args.test
    model = load_model(args.model)
    output = args.output
    X_test = pd.read_csv(testing)
    users = X_test['UserID'].values
    movies = X_test['MovieID'].values
    rating = model.predict([users, movies])

    out = []
    for i in range(len(rating)):
        ans = {"TestDataID":i+1, "Rating":rating[i][0]}
        out.append(ans)
    keys = ["TestDataID", "Rating"]
    with open(output, 'w') as f:
        o = csv.DictWriter(f, keys)
        o.writeheader()
        o.writerows(out)
    model.summary()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", help="testing data", type=str, required=True)
    parser.add_argument("--model", help="input model file", type=str, required=True)
    parser.add_argument("--output", help="output file", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
