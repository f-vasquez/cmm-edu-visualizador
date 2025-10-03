# precompute_all_keyword_similarities.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import ast

def precompute_all_keyword_similarities():
    """Precomputa TODAS las similitudes entre keywords - VERSIÓN OPTIMIZADA"""
    
    # Cargar datos reales del CSV
    df = pd.read_csv('data/capitulos_keywords_with_embeddings.csv')
    
    # Parsear embeddings del CSV real
    def parse_embedding(embedding_str):
        if isinstance(embedding_str, str):
            embedding_str = embedding_str.strip('[]')
            embedding_str = embedding_str.replace('\n', '').replace('\r', '')
            embedding_list = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
            return np.array(embedding_list)
        return np.array([])
    
    print("Parseando embeddings del CSV...")
    df['embeddings_array'] = df['keywords_embedding'].apply(parse_embedding)
    
    # Filtrar embeddings válidos
    df_valid = df[df['embeddings_array'].apply(lambda x: len(x) > 0)].copy()
    
    # Verificar dimensiones
    embedding_dims = df_valid['embeddings_array'].apply(len)
    unique_dims = embedding_dims.unique()
    print(f"Dimensiones de embeddings encontradas: {unique_dims}")
    
    if len(unique_dims) > 1:
        most_common_dim = embedding_dims.mode()[0]
        print(f"Usando dimensión más común: {most_common_dim}")
        df_valid = df_valid[df_valid['embeddings_array'].apply(len) == most_common_dim]
    
    # Agrupar por capítulo
    df_valid['capitulo_id'] = df_valid['id'].astype(int)
    
    capitulos_embeddings = {}
    capitulos_info = {}
    
    for capitulo_id, group in df_valid.groupby('capitulo_id'):
        first_row = group.iloc[0]
        capitulos_info[capitulo_id] = {
            'id': first_row['id'],
            'curso': first_row['curso'],
            'numero': first_row['numero'],
            'titulo': first_row['titulo']
        }
        
        embeddings_list = list(group['embeddings_array'])
        keywords_list = list(group['keywords'])
        
        capitulos_embeddings[capitulo_id] = {
            'embeddings': np.array(embeddings_list),
            'keywords': keywords_list
        }
        
        print(f"Capítulo {capitulo_id}: {len(keywords_list)} keywords")
    
    # Precomputar SOLO triangular superior - VERSIÓN OPTIMIZADA
    print("\nPrecomputando similitudes (VERSIÓN OPTIMIZADA)...")
    all_metrics = []
    capitulos_ids = list(capitulos_embeddings.keys())
    n_capitulos = len(capitulos_ids)
    
    total_pares = n_capitulos * (n_capitulos - 1) // 2
    pares_calculados = 0
    
    for i in range(n_capitulos):
        cap_id_i = capitulos_ids[i]
        data_i = capitulos_embeddings[cap_id_i]
        embeddings_i = data_i['embeddings']
        
        for j in range(i + 1, n_capitulos):  # ← SOLO triangular superior entre capítulos
            cap_id_j = capitulos_ids[j]
            data_j = capitulos_embeddings[cap_id_j]
            embeddings_j = data_j['embeddings']
            
            # ✅ OPTIMIZACIÓN: Calcular TODAS las métricas de forma vectorizada
            
            # 1. Cosine Similarity (vectorizado)
            cosine_matrix = cosine_similarity(embeddings_i, embeddings_j)
            cosine_similarities_flat = cosine_matrix.flatten()
            cosine_sorted = sorted(cosine_similarities_flat, reverse=True)
            
            # 2. Euclidean Distance (vectorizado)  
            euclidean_matrix = euclidean_distances(embeddings_i, embeddings_j)
            euclidean_distances_flat = euclidean_matrix.flatten()
            euclidean_sorted = sorted(euclidean_distances_flat)
            
            # 3. Dot Product (vectorizado)
            dot_matrix = np.dot(embeddings_i, embeddings_j.T)
            dot_products_flat = dot_matrix.flatten()
            dot_sorted = sorted(dot_products_flat, reverse=True)
            
            n_comparaciones = len(cosine_similarities_flat)
            
            # ✅ OPTIMIZACIÓN: Guardar solo si hay suficientes datos
            if n_comparaciones > 0:
                all_metrics.append({
                    'capitulo_i': cap_id_i,
                    'capitulo_j': cap_id_j,
                    'num_keywords_i': len(embeddings_i),
                    'num_keywords_j': len(embeddings_j),
                    'total_comparaciones': n_comparaciones,
                    'cosine_similarities': cosine_sorted,
                    'euclidean_distances': euclidean_sorted,
                    'dot_products': dot_sorted
                })
            
            pares_calculados += 1
            if pares_calculados % 10 == 0:
                print(f"Procesados {pares_calculados}/{total_pares} pares...")
                if n_comparaciones > 0:
                    print(f"   Cap {cap_id_i}({len(embeddings_i)}kw) vs Cap {cap_id_j}({len(embeddings_j)}kw) = {n_comparaciones} comparaciones")
    
    # Crear DataFrame optimizado
    print("Creando DataFrame optimizado...")
    df_metrics = pd.DataFrame(all_metrics)
    
    # ✅ OPTIMIZACIÓN: Comprimir datos para archivo más pequeño
    output_file = "data/precomputed_all_keyword_metrics.csv"
    df_metrics.to_csv(output_file, index=False, compression='gzip' if output_file.endswith('.gz') else None)
    
    print(f"✅ Precomputación OPTIMIZADA completada!")
    print(f"📊 Archivo guardado: {output_file}")
    
    return df_metrics

if __name__ == "__main__":
    df_result = precompute_all_keyword_similarities()