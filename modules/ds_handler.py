import os
from urllib.request import urlretrieve
import zipfile
from csv import DictReader
import pickle
from collections import defaultdict

procpath = 'ds/processed/{}'

def download_data():
    """
    download, extract data
    save path: ds/raw
    """
    rawpath = 'ds/raw/'
    url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
    fn = 'ml-20m.zip'
    urlretrieve(url, rawpath + fn)
    with zipfile.ZipFile(rawpath + fn) as f:
        f.extractall(rawpath)
    os.remove(rawpath + fn)

def split_save_data():
    """
    split train, val, test data and save to separate pickle file
    save path: ds/processed
    """
    ts_split = (1262271600, 1388502000, 1404140400, 1420038000)
    ratings_train, ratings_val, ratings_test = [], [], []
    with open('ds/raw/ml-20m/ratings.csv', newline='') as f:
        reader = DictReader(f)
        for row in reader:
            row = (int(row['userId']), int(row['movieId']),
                    float(row['rating']), int(row['timestamp']))
            if ts_split[0] <= row[- 1] < ts_split[1]:
                ratings_train.append(row)
            elif ts_split[1] <= row[- 1] < ts_split[2]:
                ratings_val.append(row)
            elif ts_split[2] <= row[- 1] < ts_split[3]:
                ratings_test.append(row)
    with open(procpath.format('ratings_train.pkl'), 'wb') as f:
        pickle.dump(ratings_train, f)
    with open(procpath.format('ratings_val.pkl'), 'wb') as f:
        pickle.dump(ratings_val, f)
    with open(procpath.format('ratings_test.pkl'), 'wb') as f:
        pickle.dump(ratings_test, f)
        
def save_label_encoder():
    """
    make userId, movieId to int encoder and save to pickle file
    save path: ds/processed
    """
    userIds, movieIds = set(), set()
    user_enc, movie_enc = dict(), dict()
    with open(procpath.format('ratings_train.pkl'), 'rb') as f:
        ratings_train = pickle.load(f)
    for u, m, _, _ in ratings_train:
        userIds.add(u)
        movieIds.add(m)
    for i, u in enumerate(userIds):
        user_enc[u] = i
    for i, m in enumerate(movieIds):
        movie_enc[m] = i
    with open(procpath.format('user_enc.pkl'), 'wb') as f:
        pickle.dump(user_enc, f)
    with open(procpath.format('movie_enc.pkl'), 'wb') as f:
        pickle.dump(movie_enc, f)
        
def load_mf():
    """
    load ds dataset for mf model
    if userId or movieId is not in train set, encoder returns - 1
    """
    with open(procpath.format('ratings_train.pkl'), 'rb') as f:
        ratings_train = pickle.load(f)
    with open(procpath.format('ratings_val.pkl'), 'rb') as f:
        ratings_val = pickle.load(f)
    with open(procpath.format('ratings_test.pkl'), 'rb') as f:
        ratings_test = pickle.load(f)
    with open(procpath.format('user_enc.pkl'), 'rb') as f:
        user_enc = defaultdict((lambda: -1), pickle.load(f))
    with open(procpath.format('movie_enc.pkl'), 'rb') as f:
        movie_enc = defaultdict((lambda: -1), pickle.load(f))
    return ratings_train, ratings_val, ratings_test, user_enc, movie_enc