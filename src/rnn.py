import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, KFold
from sklearn.base import clone

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size = 64, output_size = 1, batch_size = 2, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden = None):
        if hidden == None:
            hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            cell_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (hidden_0, cell_0)

        out, hidden = self.rnn(x, hidden)

        out = self.fc(out)
        out = self.sigmoid(out)

        return out, hidden
    
    def train_test(self, X_train, y_train, X_test, y_test, num_epochs = 10, learning_rate = 0.001, early_stopping_patience = 5):
        train_dataset = SequenceDataset(X_train, y_train)
        test_dataset = SequenceDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.BCELoss(reduction = 'none')
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 2, verbose=False)

        eval_data = []
        best_test_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(num_epochs):
            epoch_data = {"epoch" : epoch}

            self.train()
            train_losses = []

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output, _ = self(data)
                loss = criterion(output, target)
                batch_loss = loss.mean()

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1)
                optimizer.step()

                train_losses.append(batch_loss.item())

            train_loss = np.mean(train_losses)

            epoch_data["train_loss"] = train_loss

            self.eval()
            
            test_losses = []
            predictions = []
            true_labels = []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    output, _ = self(data)
                    loss = criterion(output, target)
                    test_losses.extend(loss.mean(dim=1).cpu().numpy())

                    predictions.append(output.cpu())
                    true_labels.append(target.cpu())
            
            test_loss = np.mean(test_losses)

            all_preds = torch.cat(predictions)
            all_labels = torch.cat(true_labels)

            all_preds_rounded = (all_preds >= 0.5).float()
            accuracy = (all_preds_rounded == all_labels).float().mean().item()

            scheduler.step(test_loss)

            epoch_data.update({"test_loss": test_loss, "test_acc": accuracy, "predictions": all_preds.numpy().squeeze(-1), "true_labels": all_labels.numpy().squeeze(-1), "learning_rate": optimizer.param_groups[0]['lr']})

            eval_data.append(epoch_data)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                best_model_state = self.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early Stopping activated at epoch #{}".format(epoch))
                    self.load_state_dict(best_model_state)
                    break
            
            print("Epoch {}: train_loss = {}, test_loss = {}, accuracy = {}, lr = {}".format(epoch, train_loss, test_loss, accuracy, optimizer.param_groups[0]['lr']))

        return eval_data


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


            



