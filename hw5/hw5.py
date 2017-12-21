from keras.models import load_model
import keras
import keras.backend as K
import pandas as pd
import numpy as np
import argparse
import h5py
import numpy as np
import pandas as pd
import sys, csv

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def getMovieCategory(movie_detail, max_movie):
    movie_dict = dict()
    movie_cat = []
    indx = 0
    for movie_id in range(max_movie):
        movie_cat.append([])
        tmp = movie_detail[movie_id].split("|")
        for cat in tmp:
            if cat in movie_dict.keys():
                movie_cat[movie_id].append(movie_dict[cat])
            else:
                movie_dict[cat] = indx
                movie_cat[movie_id].append(indx)
                indx += 1
    movie_cat_num = len(movie_dict)
    movie_cat = np.array(movie_cat)
    for movie_id in range(len(movie_cat)):
        tmp = np.zeros(movie_cat_num)
        tmp[movie_cat[movie_id]] = 1
        movie_cat[movie_id] = tmp
    return movie_cat, movie_cat_num
    

def readAdditionFile(user, movie, max_user, max_movie):
    users = pd.read_csv(user, sep='::')
    users = users.sort_values(by=['UserID']).values
    movies = pd.read_csv(movie, sep='::').values
    movie_cat = ['']*max_movie
    for movie in movies:
        movie_cat[movie[0]-1] = movie[2]
    return users, movie_cat

def getMovieCategory(movie_detail, max_movie):
    movie_dict = dict()
    movie_cat = []
    indx = 0
    for movie_id in range(max_movie):
        movie_cat.append([])
        tmp = movie_detail[movie_id].split("|")
        for cat in tmp:
            if cat in movie_dict.keys():
                movie_cat[movie_id].append(movie_dict[cat])
            else:
                movie_dict[cat] = indx
                movie_cat[movie_id].append(indx)
                indx += 1
    movie_cat_num = len(movie_dict)
    movie_cat = np.array(movie_cat)
    for movie_id in range(len(movie_cat)):
        tmp = np.zeros(movie_cat_num)
        tmp[movie_cat[movie_id]] = 1
        movie_cat[movie_id] = tmp
    return movie_cat, movie_cat_num

def main(args):
    testing = args.test
    model = load_model(args.model, custom_objects={'rmse': rmse})
    output = args.output
    X_test = pd.read_csv(testing)

    users = X_test['UserID'].values
    movies = X_test['MovieID'].values

    user_detail, movie_detail = readAdditionFile(args.user, args.movie, 6040, 4000)
    movie_cat_dict, cat_num = getMovieCategory(movie_detail, 4000)
    movie_cat = []
    for movie_id in movies:
         movie_cat.append(movie_cat_dict[movie_id-1])
    movie_cat = np.array(movie_cat)

    user_occ = []
    user_sex = []
    for user_id in users:
        user_occ.append(user_detail[user_id-1][3])
        if user_detail[user_id-1][1] == 'F':
            user_sex.append(0)
        else:
            user_sex.append(1)
    user_occ = np.array(user_occ)
    user_sex = np.array(user_sex)
    
    rating = model.predict([users, movies, movie_cat, user_occ, user_sex])

    out = []
    for i in range(len(rating)):
        ans = {"TestDataID":i+1, "Rating":rating[i][0]}
        out.append(ans)
    keys = ["TestDataID", "Rating"]
    with open(output, 'w') as f:
        o = csv.DictWriter(f, keys)
        o.writeheader()
        o.writerows(out)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", help="testing data", type=str, required=True)
    parser.add_argument("--model", help="input model file", type=str, required=True)
    parser.add_argument("--output", help="output file", type=str, required=True)
    parser.add_argument("--user", help="user file", type=str, required=True)
    parser.add_argument("--movie", help="movie file", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
