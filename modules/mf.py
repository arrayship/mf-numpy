import pickle
from collections import defaultdict
import numpy as np
import time
from copy import deepcopy
from .metrics import rmse
from csv import DictWriter

def calc_avg(train_set, user_enc, movie_enc):
    """
    make average rating dict for encoded value of userIds, movieIds
    """
    train_avg = defaultdict(lambda: [0, 0.0])
    for u, m, r, _ in train_set:
        ui, mi = user_enc[u], movie_enc[m]
        train_avg[(ui, - 1)][0] += 1
        train_avg[(ui, - 1)][1] += r
        train_avg[(- 1, mi)][0] += 1
        train_avg[(- 1, mi)][1] += r
        train_avg[(- 1, - 1)][0] += 1
        train_avg[(- 1, - 1)][1] += r
    for k, v in train_avg.items():
        train_avg[k] = v[1] / v[0]
    return train_avg

class MF():
    """
    matrix factorization (with bias, ridge regularizer) model class
    bu, bm: trainable bias vector for user, movie
    Lu, Lm: trainable latent factor matrix for user, movie
    """
    def __init__(self, n_u, n_m, f_dim, train_avg):
        """
        initialize latent factor matrices
        
        n_u, n_m: # of users, movies
        f_dim: latent factor space dimension
        train_avg: average ratings for bias and cold start
        """
        self.train_avg = train_avg
        self.bu = np.zeros(n_u)
        self.Lu = np.random.rand(n_u, f_dim)
        self.bm = np.zeros(n_m)
        self.Lm = np.random.rand(n_m, f_dim)
    
    def step(self, ui, mi, r, pc, lr):
        """
        sgd one step
        """
        r_pred = self.predict(ui, mi)
        e = self.error(r, r_pred)
        g_ui, g_mi = self.gradient(
                self.bu[ui], self.Lu[ui], self.bm[mi], self.Lm[mi], e, pc
                )
        self.update(ui, mi, g_ui, g_mi, lr)
        
    def predict(self, ui, mi):
        """
        predict score from user, movie pair
        
        ui, mi: encoded index of user, movie
        """
        if (ui == - 1) and (mi == - 1):
            return self.train_avg[(- 1, - 1)]
        elif (ui == - 1):
            return self.train_avg[(- 1, mi)]# + self.bm[mi]
        elif (mi == - 1):
            return self.train_avg[(ui, - 1)]# + self.bu[ui]
        else:
            return self.train_avg[(- 1, - 1)] +\
                    self.bu[ui] + self.bm[mi] +\
                    np.dot(self.Lu[ui], self.Lm[mi])
        
    def error(self, r, r_pred):
        """        
        r: rating of mi rated by ui
        r_pred: prediction of r
        """
        return r - r_pred
        
    def gradient(self, b_ui, l_ui, b_mi, l_mi, e, pc):
        """
        b_ui, b_mi: bias for ui, mi
        l_ui, l_mi: latent factor vector for ui, mi
        pc: regulizer penalty constant
        """
        g_ui = (((- 2) * e) + (2 * pc * b_ui),
                ((- 2) * e * l_mi) + (2 * pc * l_ui))
        g_mi = (((- 2) * e) + (2 * pc * b_mi),
                ((- 2) * e * l_ui) + (2 * pc * l_mi))
        return g_ui, g_mi
        
    def update(self, ui, mi, g_ui, g_mi, lr):
        """
        g_ui, g_mi: gradient corresponding to ui, mi
        lr: learning rate
        """
        self.bu[ui] -= (lr * g_ui[0])
        self.Lu[ui] -= (lr * g_ui[1])
        self.bm[mi] -= (lr * g_mi[0])
        self.Lm[mi] -= (lr * g_mi[1])

def train(model, train_set, val_set, user_enc, movie_enc, pc, lr, max_epoch):
    """
    train MF model
    returns model with best validation
    """
    best_model = model
    best_val_rmse = float('inf')
    train_ratings = []
    for _, _, r, _ in train_set:
        train_ratings.append(r)
    val_ratings = []
    for _, _, r, _ in val_set:
        val_ratings.append(r)
    for epoch in range(1, max_epoch + 1):
        st = time.time()
        for u, m, r, _ in train_set:
            ui, mi = user_enc[u], movie_enc[m]
            model.step(ui, mi, r, pc, lr)
        train_ratings_pred = []
        for u, m, _, _ in train_set:
            ui, mi = user_enc[u], movie_enc[m]
            train_ratings_pred.append(model.predict(ui, mi))
        train_rmse = rmse(train_ratings, train_ratings_pred)
        val_ratings_pred = []
        for u, m, _, _ in val_set:
            ui, mi = user_enc[u], movie_enc[m]
            val_ratings_pred.append(model.predict(ui, mi))
        val_rmse = rmse(val_ratings, val_ratings_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = deepcopy(model)
        et = time.time()
        print('Epoch: {:>3}, Train RMSE: {:>6.3f}, Val RMSE: {:>6.3f}, Elapsed: {:>5.1f}min'
                .format(epoch, train_rmse, val_rmse, ((et - st) / 60)))
    return best_model

def make_result(model, train_set, test_set, user_enc, movie_enc):
    """
    calculate final train, test rmse
    make predictions for test set and save to csv file
    """
    train_ratings, train_ratings_pred = [], []
    for u, m, r, _ in train_set:
        ui, mi = user_enc[u], movie_enc[m]
        train_ratings.append(r)
        train_ratings_pred.append(model.predict(ui, mi))
    train_rmse = rmse(train_ratings, train_ratings_pred)
    test_ratings, test_ratings_pred = [], []
    for u, m, r, ts in test_set:
        ui, mi = user_enc[u], movie_enc[m]
        test_ratings.append(r)
        test_ratings_pred.append(model.predict(ui, mi))
    test_rmse = rmse(test_ratings, test_ratings_pred)
    print('Train RMSE: {:.3f}, Test RMSE: {:.3f}'.format(train_rmse, test_rmse))
    with open('results.csv', 'w', newline='') as f:
        fieldnames = ['userId', 'movieId', 'predicted rating', 'timestamp']
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (u, m, _, ts), r_pred in zip(test_set, test_ratings_pred):
            writer.writerow({
                    'userId': u,
                    'movieId': m,
                    'predicted rating': r_pred,
                    'timestamp': ts
                    })