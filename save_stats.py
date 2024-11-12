import polars as pl
import statistics
from pathlib import Path


def write_stats(name, datetime, stats, data):
    filename = 'stats.csv'
    train_mean = statistics.mean([float(i) for i in stats["train_loss"]])
    val_mean = statistics.mean([float(i) for i in stats["val_loss"]])
    f1_mean = statistics.mean([float(i) for i in stats["f1"]])
    new_rows = [{
        'mode': name, 
        'datetime': datetime,
        'train_loss': float("{:.4f}".format(train_mean)),
        'val_loss': float("{:.4f}".format(val_mean)),
        'f1': float("{:.4f}".format(f1_mean)),
        'epoch': data['epoch'],
        'hidden_dim': data['hidden_dim'],
        'batch_size': data['batch_size'],
        'learning_rate': data['learning_rate'],
        'dropout': data['dropout'],
        'num_layers': data['num_layers'],
        'bidirectional': data['bidirectional'],
        'embedding': data['embedding'],
        'optimizer': data['optimizer'],
        }]
    file = Path(filename)
    if not file.exists():
        df = pl.DataFrame(new_rows[0])
    else:
        df = pl.read_csv(filename)
        new_df = pl.DataFrame(new_rows)
        df = df.extend(new_df)
    df.write_csv(filename, separator=",")
