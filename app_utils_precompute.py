import faiss
import pickle
import numpy as np
import os

def cargar_indice_faiss(method='cosine', data_dir='data'):
    index_path = os.path.join(data_dir, f'faiss_index_{method}.index')
    meta_path = os.path.join(data_dir, f'faiss_metadata_{method}.pkl')
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def cargar_dim_reduction(method='tsne', data_dir='data'):
    if method == 'tsne':
        return np.load(os.path.join(data_dir, 'tsne_embeddings.npy'))
    elif method == 'umap':
        return np.load(os.path.join(data_dir, 'umap_embeddings.npy'))
    else:
        raise ValueError('Método de reducción no soportado') 