import pandas as pd
import numpy as np
import faiss
import os
import pickle
import re
import unicodedata

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
df['etiqueta'] = df.apply(lambda row: f"Capítulo N°{row['numero']}: {row['titulo']}", axis=1)

# Métodos de similitud
methods = {
    'cosine': faiss.IndexFlatIP,
    'euclidean': faiss.IndexFlatL2,
    'dot_product': faiss.IndexFlatIP
}

def safe_metadata_string(input_str):
    if not isinstance(input_str, str):
        input_str = str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    no_accents = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Reemplaza < y > por (<) y (>)
    cleaned = no_accents.replace('<', '(<)').replace('>', '(>)')
    # Elimina otros caracteres prohibidos en nombres de archivo y HTML
    cleaned = re.sub(r'[:"/\\|?*]', '', cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "valor"
    return cleaned

def clean_keywords(keywords):
    if not isinstance(keywords, str):
        keywords = str(keywords)
    # Reemplaza saltos de línea y caracteres raros por punto y coma
    cleaned = re.sub(r'[\r\n]+', '; ', keywords)
    cleaned = cleaned.strip()
    return cleaned

for method, index_class in methods.items():
    embeddings_validos = []
    metadata_keywords = {
        "ids": [],
        "titulos": [],
        "titulos_original": [],
        "numeros": [],
        "cursos": [],
        "cursos_original": [],
        "keywords": [],
        "keywords_original": [],
        "etiquetas": [],
        "etiquetas_original": []
    }
    for idx, row in df.iterrows():
        embedding = row['embeddings_array']
        if embedding is not None and len(embedding) > 0:
            embeddings_validos.append(embedding.astype('float32'))
            metadata_keywords["ids"].append(row['id'])
            metadata_keywords["titulos"].append(safe_metadata_string(row['titulo']))
            metadata_keywords["titulos_original"].append(row['titulo'])
            metadata_keywords["numeros"].append(row['numero'])
            metadata_keywords["cursos"].append(safe_metadata_string(row['curso']))
            metadata_keywords["cursos_original"].append(row['curso'])
            metadata_keywords["keywords"].append(safe_metadata_string(row['keywords']))
            metadata_keywords["keywords_original"].append(row['keywords'])
            metadata_keywords["etiquetas"].append(safe_metadata_string(row['etiqueta']))
            metadata_keywords["etiquetas_original"].append(row['etiqueta'])
    embeddings_matrix = np.vstack(embeddings_validos).astype('float32')
    dimension = embeddings_matrix.shape[1]
    index = index_class(dimension)
    index.add(embeddings_matrix)
    # Guardar índice FAISS
    faiss.write_index(index, f"data/faiss_index_{method}.index")
    # Guardar metadatos
    with open(f"data/faiss_metadata_{method}.pkl", "wb") as f:
        pickle.dump(metadata_keywords, f)
    print(f"Guardado FAISS y metadatos para método: {method}")

print("¡Listo! Todos los índices y metadatos guardados en /data") 