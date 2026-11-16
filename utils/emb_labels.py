import pandas as pd
import numpy as np

def read_embeddings_dataset(file_path: str):
    """
    Reads the embeddings dataset from a file.
    :param file_path: path to the file where embeddings dataset is saved
    :return: a numpy ndarray of shape (num_timesteps, num_features), where num_features is the number of features in per embedding.
    """
    embeddings = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            embedding_str = parts[1]
            embedding = np.array(list(map(float, embedding_str.split(","))))
            embeddings.append(embedding)
    return np.array(embeddings)  # shape: (num_timesteps, num_features)


def align_embeddings_with_labels(embedding_path, labels_path):
    """
    Loads embeddings and labels, aligns them by dropping the first L instances 
    (due to windowing) and the last instance, and returns a unified DataFrame.
    """

    labels_df = pd.read_csv(labels_path, header=None)
    dates = labels_df.iloc[:, 0].values
    labels = labels_df.iloc[:, 1].values 
    embeddings = read_embeddings_dataset(embedding_path)
    
    offset = len(labels) - len(embeddings) - 1
    print(f"[{embedding_path}] Dropping first {offset} and last 1 instances to align data.")
    
    aligned_labels = labels[offset:-1]
    aligned_dates = dates[offset:-1]
    
    # Sanity check to ensure perfect alignment
    assert len(aligned_labels) == len(embeddings), \
        f"Mismatch: {len(aligned_labels)} labels vs {len(embeddings)} embeddings"
    
    feature_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feature_cols)
    
    # Insert label and date at the front of the DataFrame
    df.insert(0, 'label', aligned_labels) 
    df.insert(0, 'date', aligned_dates)   
    
    return df