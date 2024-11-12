import torch.nn as nn
import torch
import math
from torchcrf import CRF


class LSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers=5, dropout=0.5, bidirectional=True, embedding_matrix=None):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # Freeze embeddings to retain GloVe weights
        # use nn.Embedding.from_pretrained(embedding) if you want to initialize from existing embeddings
        # read more here https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, tagset_size)
        self.dropout = nn.Dropout(0.3)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, token_ids, tags=None):
        embeddings = self.embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        rnn_out, _ = self.lstm(embeddings)  # (batch_size, seql_len, hidden_dim)
        rnn_out = self.dropout(rnn_out)
        outputs = self.fc(rnn_out)  # (batch_size, seq_len, tagset_size)
        outputs = outputs.float()
        # return outputs
        if tags is not None:
            return -self.crf(outputs, tags)
        else:
            return self.crf.decode(outputs)
 
 
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers=1, bidirectional=False, embedding_matrix=None):
        super(GRU, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # Freeze embeddings to retain GloVe weights
        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Fully connected layer for classification over token labels
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
        
    def forward(self, x, tags=None):
        # Embed the input tokens
        x = self.embedding(x)
        
        # Forward propagate through the GRU layer
        gru_out, _ = self.gru(x)
        
        # Pass the GRU output through the fully connected layer
        outputs = self.fc(gru_out)
        outputs = outputs.float()
        # return outputs
        if tags is not None:
            return -self.crf(outputs, tags)
        else:
            return self.crf.decode(outputs)
        # return out  # Shape: (batch_size, seq_length, num_labels)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, tagset_size, num_filters=100, kernel_size=3, dropout=0.5):
        super(CNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
        
    def forward(self, x, tags=None):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.relu(x)        
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_length, num_filters)
        x = self.fc(x)           # Shape: (batch_size, seq_length, num_labels)
        outputs = x.float()
        # return outputs
        if tags is not None:
            return -self.crf(outputs, tags)
        else:
            return self.crf.decode(outputs)
        return x  # Output logits for each token

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, dropout=0.5):
        super(MLP, self).__init__()        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, math.ceil(hidden_dim/2))
        self.fc3 = nn.Linear(math.ceil(hidden_dim/2), tagset_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(tagset_size, batch_first=True)
        
    def forward(self, x, tags=None):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        outputs = x.float()
        # return outputs
        if tags is not None:
            return -self.crf(outputs, tags)
        else:
            return self.crf.decode(outputs)
        # return x 