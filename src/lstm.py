import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone
import logging
import sys
from datetime import datetime

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.clf = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = self.clf(lstm_output[:, -1, :])
        return output
    
class SklearnLSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size = 4, hidden_size = 50, sub_sequence_length = 5, scaler = None, lr = 0.001, num_epochs = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_sequence_length = sub_sequence_length
        self.lr = lr
        self.num_epochs = num_epochs
        self.scaler = scaler
    
    def create_train_sequences(self, X):
        sub_sequences = []
        sub_sequence_labels = []

        X_new = X.copy()

        if self.scaler is not None:
            X_concat = pd.concat(X)
            self.scaler.fit(X_concat)
            X_new = [self.scaler.transform(frame) for frame in X]
        else:
            X_new = [frame.to_numpy() for frame in X]

        for seq_idx in range(len(X)):
            X_num = X_new[seq_idx]
            for i in range(self.sub_sequence_length-1, X_num.shape[0]):
                sub_seq = X_num[i-self.sub_sequence_length+1:i+1]
                sub_sequences.append(sub_seq)
                sub_sequence_labels.append(seq_idx)
        return np.array(sub_sequences), np.array(sub_sequence_labels)
    
    def create_test_sequences(self, X):
        sub_sequences = []
        sub_sequence_labels = []

        X_new = X.copy()

        if self.scaler is not None:
            X_new = [self.scaler.transform(frame) for frame in X]
        else:
            X_new = [frame.to_numpy() for frame in X]

        for seq_idx in range(len(X)):
            X_num = X_new[seq_idx]
            for i in range(self.sub_sequence_length-1, X_num.shape[0]):
                sub_seq = X_num[i-self.sub_sequence_length+1:i+1]
                sub_sequences.append(sub_seq)
                sub_sequence_labels.append(seq_idx)
        return np.array(sub_sequences), np.array(sub_sequence_labels)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_sub, seq_idxs = self.create_train_sequences(X)
        y_sub = [y[idx] for idx in seq_idxs]

        self.input_size = X[0].shape[1]

        X_tensor = torch.FloatTensor(X_sub)
        y_tensor = torch.LongTensor(y_sub)

        num_classes = len(np.unique(y_sub))
        self.model = LSTM(self.input_size, self.hidden_size, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

        self.model.train()
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)

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

def sequence_prediction_scorer(model, X, y):
    y_pred, test_idx = model.predict(X)
    y_true = [y[idx] for idx in test_idx]
    acc = accuracy_score(y_true, y_pred)
    return acc