import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging
import sys
from datetime import datetime

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0.0, recurrent_dropout = 0.0, bidirectional = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.recurrent_dropout = nn.Dropout(recurrent_dropout) if (num_layers == 1 and not bidirectional) else nn.Dropout(0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout if num_layers > 1 else 0, bidirectional = bidirectional)
        self.coef = 2 if bidirectional else 1
        self.clf = nn.Linear(hidden_size * self.coef, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.coef, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.coef, x.size(0), self.hidden_size).to(x.device)

        if self.num_layers > 1 or self.bidirectional:
            output, (h_n, c_n) = self.lstm(x, (h0, c0))
            last_output = output[:, -1, :]
        else:
            outputs = []
            h_t = h0[0]
            c_t = c0[0]

            

            for t in range(x.size(1)):
                # print(f"Step {t}: h_t before dropout: {torch.isnan(h_t).any()}")
                h_dropped = self.recurrent_dropout(h_t)
                # print(f"Step {t}: h_t after dropout: {torch.isnan(h_dropped).any()}")
                _, (h_t, c_t) = self.lstm(x[:, t:t+1, :], (h_dropped.unsqueeze(0), c_t.unsqueeze(0)))
                # print(f"Step {t}: h_t after lstm: {torch.isnan(h_t).any()}")

                h_t = h_t.squeeze(0)
                c_t = c_t.squeeze(0)
                outputs.append(h_t)

            outputs = torch.stack(outputs, dim = 1)
            last_output = outputs[:, -1, :]

        output = self.clf(last_output)
        return output

"""
Wrapper for pytorch LSTM model
Accommodates choosing different featuresets and pca features
Includes functions for generating sub_sequences with length sub_sequence_length from datasets
Follows sklearn classifier standards in order to work with Gridsearch
Takes in array of data, with elements (n_samples, n_features)
"""
class SklearnLSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, features = [], scaler = None, pca_features = [], pca = None, hidden_size = 50, sub_sequence_length = 5, batch_size = 1, lr = 0.001, num_epochs = 10, num_layers = 1, dropout = 0, recurrent_dropout = 0, bidirectional = False, optimizer = 'adam', momentum = 0.9, weight_init = 'xavier', l2_lambda = 0):
        self.input_size = (len(features) + pca.n_components) if pca is not None else len(features)
        self.hidden_size = hidden_size
        self.sub_sequence_length = sub_sequence_length
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.features = features
        self.pca_features = pca_features
        # self.scaler = clone(scaler) if scaler else None
        # self.pca = clone(pca) if pca else None
        self.scaler = scaler
        self.pca = pca
        self.num_layers = num_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_init = weight_init
        self.l2_lambda = l2_lambda
    
    def create_train_sequences(self, X):
        sub_sequences = []
        sub_sequence_labels = []

        pca_idxs = [list(X[0].columns).index(feat) for feat in self.pca_features]
        feat_idxs = [list(X[0].columns).index(feat) for feat in self.features]

        X_new = X.copy()

        if self.scaler is not None:
            X_concat = pd.concat(X)
            self.scaler.fit(X_concat)
            X_new = [self.scaler.transform(frame) for frame in X]
        else:
            X_new = [frame.to_numpy() for frame in X]
        
        if self.pca is not None and len(pca_idxs) > 0:
            X_concat_pca = np.concatenate(X_new)[:, pca_idxs]
            self.pca.fit(X_concat_pca)
            for i in range(len(X_new)):
                X_cur = X_new.pop(0)
                X_cur_feats = X_cur[:, feat_idxs]
                X_cur_pca = X_cur[:, pca_idxs]
                X_cur_pca = self.pca.transform(X_cur_pca)
                X_temp = np.append(X_cur_feats, X_cur_pca, 1)
                X_new.append(X_temp)

        for seq_idx in range(len(X_new)):
            X_num = X_new[seq_idx]
            for i in range(self.sub_sequence_length-1, X_num.shape[0]):
                sub_seq = X_num[i-self.sub_sequence_length+1:i+1]
                sub_sequences.append(sub_seq)
                sub_sequence_labels.append(seq_idx)
        return np.array(sub_sequences), np.array(sub_sequence_labels)
    
    def create_test_sequences(self, X):
        sub_sequences = []
        sub_sequence_labels = []

        all_feats = list(set(self.features + self.pca_features))
        pca_idxs = [list(X[0].columns).index(feat) for feat in self.pca_features]
        feat_idxs = [list(X[0].columns).index(feat) for feat in self.features]

        X_new = X.copy()

        if self.scaler is not None:
            X_new = [self.scaler.transform(frame) for frame in X]
        else:
            X_new = [frame.to_numpy() for frame in X]
        
        if self.pca is not None and len(pca_idxs) > 0:
            for i in range(len(X_new)):
                X_cur = X_new.pop(0)
                X_cur_feats = X_cur[:, feat_idxs]
                X_cur_pca = X_cur[:, pca_idxs]
                X_cur_pca = self.pca.transform(X_cur_pca)
                X_temp = np.append(X_cur_feats, X_cur_pca, 1)
                X_new.append(X_temp)

        for seq_idx in range(len(X_new)):
            X_num = X_new[seq_idx]
            for i in range(self.sub_sequence_length-1, X_num.shape[0]):
                sub_seq = X_num[i-self.sub_sequence_length+1:i+1]
                sub_sequences.append(sub_seq)
                sub_sequence_labels.append(seq_idx)
        return np.array(sub_sequences), np.array(sub_sequence_labels)

    """
    X: arr = [pd.DataFrame.shape = (n_samples, n_features)]
    y: arr = [pd.DataFrame.shape = (n_samples, seat)]
    """
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_sub, seq_idxs = self.create_train_sequences(X)
        y_sub = [y[idx] for idx in seq_idxs]

        self.input_size = X_sub.shape[2]

        X_tensor = torch.FloatTensor(X_sub)
        y_tensor = torch.LongTensor(y_sub)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        num_classes = len(np.unique(y_sub))
        self.model = LSTM(self.input_size, self.hidden_size, num_classes, self.num_layers, self.dropout, self.recurrent_dropout, self.bidirectional)

        if self.weight_init == 'xavier':
            torch.nn.init.xavier_uniform_(self.model.lstm.weight_ih_l0)

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.l2_lambda)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum = self.momentum, weight_decay = self.l2_lambda)


        criterion = nn.CrossEntropyLoss()
        

        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X_sub, seq_idxs = self.create_test_sequences(X)
        X_tensor = torch.FloatTensor(X_sub)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, sub_preds = torch.max(outputs.data, 1)
        return sub_preds.numpy(), seq_idxs
    
    def predict_proba(self, X):
        X_sub, seq_idxs = self.create_test_sequences(X)
        X_tensor = torch.FloatTensor(X_sub)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs, seq_idxs
    
    def KFold(self, X, y, n_splits = 5):
        kf = KFold(n_splits=n_splits)

        scores = []
        
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train = [X[idx] for idx in train_index]
            X_test = [X[idx] for idx in test_index]
            y_train = [y[idx] for idx in train_index]
            y_test = [y[idx] for idx in test_index]

            model = clone(self)

            model.fit(X_train, y_train)

            y_pred, test_idx = model.predict(X_test)

            y_true = [y_test[idx].iloc[0] for idx in test_idx]

            acc = accuracy_score(y_true, y_pred)
            scores.append(acc)
        
        return scores

"""
Scoring function to be used in Gridsearch
Generates tags for sequences based on original index of sequence, uses those for accuracy
model: trained LSTM model used to predict labels
X: array of test data of shape (n_samples, n_features)
y: array of test data of shape (n_samples,)
"""
def sequence_prediction_scorer(model, X, y):
    y_pred, test_idx = model.predict(X)
    y_true = [y[idx] for idx in test_idx]
    acc = accuracy_score(y_true, y_pred)
    return acc