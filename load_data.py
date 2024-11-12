import polars as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

def processData(file_path):
    df = pl.read_csv(file_path)
    df = df.drop_nulls()
    x = df['utterances'].to_list()
    y = df.get_column("IOB Slot tags", default=[])
    return x, y

def constructDataset(utterances, tags=[]):
    data = []
    for sentence in utterances:
        tokens = sentence.split()
        data.append({
            "sentence": tokens
        })
    for index, sentTag in enumerate(tags):
        tokenTags = sentTag.split()
        data[index]["labels"] = tokenTags
    return data

def splitData(X, y, split):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=42)
    return x_train, x_test, y_train, y_test

# create training datasets
def createTagSets(filepath, split=0.2):
    utterances, tags = processData(filepath)
    x_train, x_val, y_train, y_val = splitData(utterances, tags, split)
    trainData = constructDataset(x_train, y_train)
    valData = constructDataset(x_val, y_val)
    train_dataset = POSDataset(trainData, training=True)
    val_dataset = POSDataset(valData, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
    return train_dataset, val_dataset

def createGridSearchSets(filepath, split=0.005):
    utterances, tags = processData(filepath)
    x_train, x_val, y_train, y_val = splitData(utterances, tags, split)
    trainData = constructDataset(x_train, y_train)
    valData = constructDataset(x_val, y_val)
    return trainData, valData

# create testing datasets
def createTestData(filepath, train_dataset):
    utterances, tags = processData(filepath)
    testData = constructDataset(utterances)
    test_dataset = POSDataset(testData, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
    return test_dataset, utterances

# define dataset
class POSDataset(Dataset):
    def __init__(self, data, token_vocab=None, tag_vocab=None, training=True):

        # Create vocabularies if training
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}
            self.all_tags = []

            # build vocab from training data
            for item in data:
                for token in item['sentence']:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
                for tag in item['labels']:
                    if tag not in self.tag_vocab:
                        self.tag_vocab[tag] = len(self.tag_vocab)
                    self.all_tags.append(self.tag_vocab[tag])
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab

        # Convert sentences and tags to integer IDs during initialization
        self.corpus_token_ids = []
        self.corpus_tag_ids = []
        for item in data:
            token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in item['sentence']]
            tag_ids = [self.tag_vocab[tag] for tag in item['labels']] if item.get("labels") else []
            self.corpus_token_ids.append(torch.tensor(token_ids))
            self.corpus_tag_ids.append(torch.tensor(tag_ids))

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]
    
    def get_tag_vocab_inverse(self):
        return {v: k for k, v in self.tag_vocab.items()}
