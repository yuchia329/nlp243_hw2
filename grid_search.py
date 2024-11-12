from model import LSTM, GRU, CNN, MLP
from load_data import processData, createTagSets, createGridSearchSets
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

EMBEDDING_DIM = 200
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class POSDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        """
        Args:
            X: List of tokenized input sequences (list of words).
            y: List of IOB tag sequences corresponding to X (list of tags).
            tokenizer: Tokenizer to convert words to indices.
            max_len: Maximum length of sequences (for padding).
        """
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentence = self.X[idx]
        tags = self.y[idx]

        # Convert words to indices using tokenizer
        input_ids = [self.tokenizer.get(word, self.tokenizer["<unk>"]) for word in sentence]
        input_ids = input_ids[:self.max_len]  # Truncate if longer than max_len
        input_ids = input_ids + [self.tokenizer["<pad>"]] * (self.max_len - len(input_ids))  # Padding

        # Convert tags to indices
        tag_ids = [self.tokenizer["<pad>"]] * self.max_len
        for i, tag in enumerate(tags[:self.max_len]):
            tag_ids[i] = tag

        return torch.tensor(input_ids), torch.tensor(tag_ids)

class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=128, hidden_size=64, output_size=10, num_layers=1, dropout=0.5, lr=1e-3, epochs=10, batch_size=32, max_len=100, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device

    def fit(self, X, y):
        # Initialize the model
        model = LSTM(self.input_size, self.hidden_size, self.output_size, self.num_layers, self.dropout)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index for loss computation

        # Convert data into DataLoader format
        train_dataset = POSDataset(X, y, tokenizer=self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, self.output_size), labels.view(-1))  # Flatten for CrossEntropyLoss
                loss.backward()
                optimizer.step()

        self.model = model  # Store the trained model
        return self

    def predict(self, X):
        # Predict method (for evaluation)
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.long).to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 2)  # Get the predicted label for each token
        return predicted.numpy()

    def score(self, X, y):
        # Calculate F1-score (weighted average) for evaluation
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='weighted')

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer



def selectModel(name, train_dataset):
    if name == "lstm":
        model = LSTMWrapper(input_size=EMBEDDING_DIM, output_size=len(train_dataset.tag_vocab), device=device)
        model.set_tokenizer(train_dataset.token_vocab)
    elif name == "gru":
        model = GRU(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            num_layers=5, 
            bidirectional=False)
    elif name == "cnn":
        model = CNN(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            num_filters=100,
            kernel_size=3,
            dropout=0.5)
    elif name == "mlp":
        model = MLP(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            dropout=0.5)
    else:
        model = GRU(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            num_layers=5, 
            bidirectional=False)
    # model = model.to(device)
    return model

def main():
     
    param_grid = {
        'hidden_size': [64, 128],
        'num_layers': [1, 2],
        'dropout': [0.3, 0.5],
        'lr': [1e-3, 1e-4],
        'batch_size': [32, 64],
        'epochs': [10, 20, 30, 40]
    }
    train_dataset, val_dataset = createTagSets('data/hw2_train.csv', 0.05)
    trainData, valData = createGridSearchSets('data/hw2_train.csv')
    x = [item['sentence'] for item in trainData]
    y = [item['labels'] for item in trainData]
    
    model = selectModel('lstm', train_dataset)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)
    grid_search.fit(x, y)
    print("Best hyperparameters:", grid_search.best_params_)

if __name__ == "__main__":
    main()