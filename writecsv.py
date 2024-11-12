import polars as pl

def writePrediction(utterances, prediction, dt_string, output_file=None):
    tags = []
    for utt in utterances:
        n = len(utt.split())
        tags.append(" ".join(prediction[:n]))
        prediction = prediction[n:]
    
    df = pl.DataFrame(tags)
    df.columns = ["IOB Slot tags"]
    df = df.with_columns(pl.Series("ID", range(1, len(df) + 1)))
    df = df.select(["ID"] + df.columns[:-1])
    if output_file is not None:
        filename = output_file
    else:
        filename = f"result/result_{dt_string}.csv"
    print(filename)
    df.write_csv(filename, separator=",")
    
    
    