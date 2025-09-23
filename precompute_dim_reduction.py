import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import pickle

# Cargar datos
csv_path = 'data/capitulos_keywords_with_embeddings.csv'
df = pd.read_csv(csv_path)

def parse_embedding(embedding_str):
    if isinstance(embedding_str, str):
        embedding_str = embedding_str.strip('[]')
        embedding_list = [float(x.strip()) for x in embedding_str.split(',')]
        return np.array(embedding_list)
    return np.array(embedding_str)

df['embeddings_array'] = df['keywords_embedding'].apply(parse_embedding)
df = df.dropna(subset=['embeddings_array'])

embeddings_matrix = np.vstack(df['embeddings_array'].values)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_matrix)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_matrix)-1))
tsne_embeddings = tsne.fit_transform(embeddings_scaled)
with open("data/tsne_embeddings.npy", "wb") as f:
    np.save(f, tsne_embeddings)

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_reducer.fit_transform(embeddings_scaled)
with open("data/umap_embeddings.npy", "wb") as f:
    np.save(f, umap_embeddings)

print("¡Listo! Embeddings t-SNE y UMAP guardados en /data") 