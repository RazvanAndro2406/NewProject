from cuml.manifold import UMAP 
import matplotlib.pyplot as plt

import sys
sys.path.append('..') 
from utils.emb_labels import align_embeddings_with_labels, read_embeddings_dataset

def generate_umap_plot(embedding_path, labels_path, n_neighbors=15, n_components=2, min_dist=0.1):
    """
    computes UMAP on the GPU, and plots the first 2 dimensions.
    """
    print(f"Processing: {embedding_path} with {labels_path}")
    
    aligned_df = align_embeddings_with_labels(embedding_path, labels_path)
    
    features = aligned_df.drop(columns=['date', 'label']).values
    labels = aligned_df['label'].values
    
    print(f"n_components={n_components}, min_dist={min_dist})...")
    reducer = UMAP(
        n_components=n_components, 
        metric='cosine', 
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        random_state=42 
    )
    
    embedding_reduced = reducer.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(
        embedding_reduced[:, 0], 
        embedding_reduced[:, 1], 
        c=labels,           
        cmap='coolwarm',    
        s=1,                
        alpha=0.5           
    )
    
    file_name = embedding_path.split('/')[-1].split('\\')[-1]
    
    plt.title(f"GPU UMAP: {file_name} (min_dist={min_dist})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    
    plt.show()