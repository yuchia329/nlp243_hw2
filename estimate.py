import polars as pl
import pandas as pd
import argparse

# Load the previous best result and the new result
def token_level_accuracy(best_result, new_result):
    assert best_result.shape == new_result.shape, "Files do not match in size!"

    # Calculate token-level accuracy
    matching_tokens = (best_result['IOB Slot tags'] == new_result['IOB Slot tags']).sum()
    total_tokens = best_result.shape[0]

    # Token-level accuracy
    accuracy = matching_tokens / total_tokens
    print(f"Token-level accuracy: {accuracy:.4f}")
    
def sequence_level_accuracy(best_result, new_result):
    best_groups = best_result.groupby("ID")['IOB Slot tags'].apply(list)
    new_groups = new_result.groupby("ID")['IOB Slot tags'].apply(list)

    # Compare sequences
    matching_sequences = sum(best_groups == new_groups)
    sequence_accuracy = matching_sequences / len(best_groups)

    print(f"Sequence-level accuracy: {sequence_accuracy:.4f}")
    
def f1_score(best_result, new_result):
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    # Extract the unique slot tags from the 'IOB Slot tags' column
    best_tags = best_result['IOB Slot tags'].to_list()
    new_tags = new_result['IOB Slot tags'].to_list()

    # Flatten the tags into a single list of token-wise tags for comparison
    best_flat = [tag for sublist in best_tags for tag in sublist.split()]
    new_flat = [tag for sublist in new_tags for tag in sublist.split()]

    # Calculate the weighted F1 score
    f1 = f1_score(best_flat, new_flat, average='weighted')  # 'macro' could be used for unweighted average
    print(f"Weighted F1 score: {f1:.4f}")
    print(classification_report(best_flat, new_flat))
    conf_matrix = confusion_matrix(best_flat, new_flat)
    print("Confusion Matrix:\n", conf_matrix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=str, default='result/result_2024-11-09-22:55:40.csv', nargs='?',
                        help="Path to the training result CSV file")
    args = parser.parse_args()
    result_file = args.result_file
    best_result = pl.read_csv("result/result_2024-11-09-22:55:40.csv")
    new_result = pl.read_csv(result_file)
    best_result1 = pd.read_csv("result/result_2024-11-09-22:55:40.csv")
    new_result1 = pd.read_csv(result_file)
    token_level_accuracy(best_result, new_result)
    sequence_level_accuracy(best_result1, new_result1)
    f1_score(best_result, new_result)

if __name__ == "__main__":
    main()