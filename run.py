import numpy as np
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from load_data import createTagSets, createTestData
from embedding import load_glove_embeddings
from writecsv import writePrediction
from model import LSTM, GRU, CNN, MLP
from save_stats import write_stats
from datetime import datetime

def setSeed():
    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def selectModel(name, train_dataset, embedding_matrix=None):
    if name == "lstm":
        model = LSTM(
            vocab_size=len(train_dataset.token_vocab),
            tagset_size=len(train_dataset.tag_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            embedding_matrix=embedding_matrix)
    elif name == "gru":
        model = GRU(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            num_layers=NUM_LAYERS, 
            bidirectional=BIDIRECTIONAL,
            embedding_matrix=embedding_matrix)
    elif name == "cnn":
        model = CNN(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            num_filters=100,
            kernel_size=3,
            dropout=DROPOUT)
    elif name == "mlp":
        model = MLP(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            dropout=DROPOUT)
    else:
        model = GRU(
            vocab_size=len(train_dataset.token_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            tagset_size=len(train_dataset.tag_vocab),
            num_layers=NUM_LAYERS, 
            bidirectional=BIDIRECTIONAL)
    return model

def selectOptimizer(model, name):
    if name =='adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    elif name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return optimizer

def selectEmbedding(token_vocab, name):
    if name == 'glove_42b_300':
        glove_embeddings, dimension = load_glove_embeddings()
        embedding_matrix = word2Index(token_vocab, glove_embeddings, dimension)
        global EMBEDDING_DIM
        EMBEDDING_DIM = dimension
        return embedding_matrix, EMBEDDING_DIM
    else:
        return None, EMBEDDING_DIM

def train(model, train_loader, val_loader, device, loss_fn, optimizer, train_dataset):
    # Training Loop
    stats = {'train_loss': [1], 'val_loss': [1], 'f1': [1]}
    for index, epoch in enumerate(range(NUM_EPOCHS)):
        # Training
        model.train()
        total_train_loss = 0
        for token_ids, tag_ids in train_loader:
            token_ids = token_ids.to(device)
            tag_ids = tag_ids.to(device)

            optimizer.zero_grad()

            loss = model(token_ids, tag_ids)  # (batch_size, seq_len, tagset_size)
            # loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_train_loss / len(train_loader)}")
        # Validation
        # model.eval()
        # total_val_loss = 0
        # all_predictions = []
        # all_tags = []

        # with torch.no_grad():
        #     for token_ids, tag_ids in val_loader:
        #         # print(token_ids)
        #         token_ids = token_ids.to(device)
        #         tag_ids = tag_ids.to(device)

        #         outputs = model(token_ids, tag_ids)  # (batch_size, seq_len, tagset_size)
        #         outputs = outputs.view(-1, outputs.shape[-1])
        #         tag_ids = tag_ids.view(-1)
        #         loss = loss_fn(outputs, tag_ids)
        #         total_val_loss += loss.item()

        #         predictions = outputs.argmax(dim=1)
        #         mask = tag_ids != train_dataset.tag_vocab['<PAD>']

        #         all_predictions.extend(predictions[mask].tolist())
        #         all_tags.extend(tag_ids[mask].tolist())

        # # compute train and val loss
        # train_loss = total_train_loss / len(train_loader)
        # val_loss = total_val_loss / len(val_loader)

        # # Calculate F1 score
        # f1 = f1_score(all_tags, all_predictions, average='weighted')
        # if NUM_EPOCHS - index < 6:
        #     stats['train_loss'].append("{:.4f}".format(train_loss))
        #     stats['val_loss'].append("{:.4f}".format(val_loss))
        #     stats['f1'].append("{:.4f}".format(f1))
        # print(f'{epoch = } | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')
    return stats
    
def eval(model, test_loader, device, train_dataset):
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for token_ids, tag_ids in test_loader:
            token_ids = token_ids.to(device)
            predictions = model(token_ids)  # (batch_size, seq_len, tagset_size)
            # print(predictions)
            predictions = torch.tensor(predictions)
            # outputs = outputs.view(-1, outputs.shape[-1])
            # token_ids = token_ids.view(-1)
            # predictions = outputs.argmax(dim=1)
            mask = token_ids != train_dataset.tag_vocab['<PAD>']
            tag_vocab_inverse = train_dataset.get_tag_vocab_inverse()
            for label in predictions[mask].tolist():
                all_predictions.append(tag_vocab_inverse[label])
    return all_predictions

def makeStats(stats):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H:%M:%S")
    data = {
        'epoch': NUM_EPOCHS,
        'hidden_dim': HIDDEN_DIM,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'dropout': DROPOUT,
        'num_layers': NUM_LAYERS,
        'bidirectional': BIDIRECTIONAL,
        'embedding': EMBEDDING,
        'optimizer': OPTIMIZER,
        }
    write_stats(MODEL_NAME, dt_string, stats, data)
    return dt_string

def word2Index(token_vocab, glove_embeddings, embedding_dim):    
    embedding_matrix = np.zeros((len(token_vocab), embedding_dim))
    for word, idx in token_vocab.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, default='data/hw2_train.csv', nargs='?',
                        help="Path to the training data CSV file")
    parser.add_argument("test_file", type=str, default='data/hw2_test.csv', nargs='?',
                        help="Path to the test data CSV file")
    parser.add_argument("output_file", type=str, default='result/result.csv', nargs='?',
                        help="Path to the output CSV file")
    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    output_file = args.output_file
    train_dataset, val_dataset = createTagSets(train_file)
    embedding_matrix, dimension = selectEmbedding(train_dataset.token_vocab, EMBEDDING)
    
    # collate token_ids and tag_ids to make mini-batches
    def collate_fn(batch):
        # batch: [(token_ids, tag_ids), (token_ids, tag_ids), ...]

        # Separate sentences and tags
        token_ids = [item[0] for item in batch]
        tag_ids = [item[1] for item in batch]

        # Pad sequences
        sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab['<PAD>'])
        # sentences_pad.size()  (batch_size, seq_len)
        tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab['<PAD>'])
        # tags_pad.size()  (batch_size, seq_len)
        return sentences_padded, tags_padded
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = selectModel(MODEL_NAME, train_dataset, embedding_matrix)
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab['<PAD>'])
    optimizer = selectOptimizer(model, OPTIMIZER)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model = model.to(device)
    stats = train(model, train_loader, val_loader, device, loss_fn, optimizer, train_dataset)
    test_dataset, utterances = createTestData(test_file, train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    all_predictions = eval(model, test_loader, device, train_dataset)
    dt_string = makeStats(stats)
    writePrediction(utterances, all_predictions, dt_string, output_file)


if __name__ == "__main__":
    MODEL_NAME='gru'
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    DROPOUT = 0.3
    NUM_LAYERS=5
    BIDIRECTIONAL=True
    EMBEDDING = "self"
    OPTIMIZER='adam'
    # import itertools
    # embedding_dim_list = [100, 300]
    # hidden_dim_list = [128]
    # batch_size_list = [32, 64]
    # learning_rate_list = [0.001, 0.0001]
    # num_epoch_list = [50]
    # dropout_list = [0.3, 0.4]
    # num_layers_list = [1, 3]
    # bidirectional_list = [True, False]
    # embedding_list = ['glove_42b_300']#,'self']
    # optimizer_list = ['adam', 'adamw']
    # combinations = list(itertools.product(embedding_dim_list, hidden_dim_list, batch_size_list, learning_rate_list, num_epoch_list, dropout_list, num_layers_list, bidirectional_list, embedding_list, optimizer_list))
    # for index, combination in enumerate(combinations):
    #     print(combination)
    #     MODEL_NAME ='cnn'
    #     OPTIMIZER = combination[9]
    #     EMBEDDING_DIM = combination[0]
    #     HIDDEN_DIM = combination[1]
    #     BATCH_SIZE = combination[2]
    #     LEARNING_RATE = combination[3]
    #     NUM_EPOCHS = combination[4]
    #     DROPOUT = combination[5]
    #     NUM_LAYERS=combination[6]
    #     BIDIRECTIONAL=combination[7]
    #     EMBEDDING = combination[8]
    main()
        