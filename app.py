import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import time
from app_utils_precompute import cargar_indice_faiss, cargar_dim_reduction
from typing import List, Dict, Any, Tuple
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import unicodedata
import re
import sys

# Cargar variables de entorno
load_dotenv()

# Configuración para evitar warnings de OpenMP en macOS
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

def ordenar_cursos_personalizado(cursos_list):
    """
    Ordena los cursos en el orden específico: Primero Básico, Segundo Básico, etc.
    Funciona con "Básico" (con tilde) y "Basico" (sin tilde).
    """
    # Definir el orden específico con ambos formatos
    orden_especifico_con_tilde = [
        'Primero Básico',
        'Segundo Básico', 
        'Tercero Básico',
        'Cuarto Básico',
        'Quinto Básico',
        'Sexto Básico'
    ]
    
    orden_especifico_sin_tilde = [
        'Primero Basico',
        'Segundo Basico', 
        'Tercero Basico',
        'Cuarto Basico',
        'Quinto Basico',
        'Sexto Basico'
    ]
    
    # Crear un mapeo de cursos a su posición en el orden específico
    orden_mapping = {}
    
    # Mapear cursos con tilde
    for i, curso in enumerate(orden_especifico_con_tilde):
        orden_mapping[curso] = i
    
    # Mapear cursos sin tilde
    for i, curso in enumerate(orden_especifico_sin_tilde):
        orden_mapping[curso] = i
    
    # Ordenar los cursos según el mapeo
    def get_order(cursos_list):
        cursos_ordenados = []
        cursos_restantes = []
        
        for curso in cursos_list:
            if curso in orden_mapping:
                cursos_ordenados.append((curso, orden_mapping[curso]))
            else:
                cursos_restantes.append(curso)
        
        # Ordenar por posición y extraer solo los nombres
        cursos_ordenados.sort(key=lambda x: x[1])
        return [curso[0] for curso in cursos_ordenados] + cursos_restantes
    
    return get_order(cursos_list)

# Configurar cliente OpenAI
@st.cache_resource
def get_openai_client():
    """Inicializar cliente OpenAI"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ No se encontró OPENAI_API_KEY en las variables de entorno. Crea un archivo .env con tu API key.")
        return None
    return OpenAI(api_key=api_key)

def obtener_embedding_pregunta(texto: str, client) -> np.ndarray:
    """
    Obtiene el embedding vectorial para una pregunta usando OpenAI.
    
    Args:
        texto (str): La pregunta a convertir en embedding.
        client: Cliente OpenAI
        
    Returns:
        np.ndarray: El vector de embedding de la pregunta.
    """
    try:
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        embedding = np.array(response.data[0].embedding).astype("float32")
        return embedding
    except Exception as e:
        st.error(f"Error al obtener embedding: {e}")
        return None

# Eliminar funciones: construir_indice_faiss, reduce_dimensions
# y cualquier cálculo de t-SNE/UMAP/FAISS salvo la query

def load_data():
    """Cargar y procesar los datos del CSV"""
    try:
        df = pd.read_csv('data/capitulos_keywords_with_embeddings.csv')
        def parse_embedding(embedding_str):
            if isinstance(embedding_str, str):
                embedding_str = embedding_str.strip('[]')
                embedding_list = [float(x.strip()) for x in embedding_str.split(',')]
                return np.array(embedding_list)
            return np.array(embedding_str)
        df['embeddings_array'] = df['keywords_embedding'].apply(parse_embedding)
        df = df.dropna(subset=['embeddings_array'])
        df['etiqueta'] = df.apply(lambda row: f"Capítulo N°{row['numero']}: {row['titulo']}", axis=1)
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

# --- Embeddings Tab ---
def embeddings_tab():
    st.markdown('<h1 class="main-header">📊 Visualizador de Embeddings</h1>', unsafe_allow_html=True)
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el archivo CSV esté en la carpeta 'data/'.")
        return
    # Configuración dentro de la pestaña
    st.subheader("Configuración de visualización")
    cursos_disponibles = ['Todos'] + ordenar_cursos_personalizado(df['curso'].unique().tolist())
    curso_seleccionado = st.selectbox("Seleccionar Curso:", cursos_disponibles, key="embeddings_curso")
    color_by = st.selectbox(
        "Colorear por:",
        ['curso', 'numero+curso', 'numero'],
        format_func=lambda x: {
            'curso': 'Curso',
            'numero+curso': 'Número de Capítulo + Curso',
            'numero': 'Número de Capítulo'
        }[x],
        key="embeddings_color_by"
    )
    metodo = 'tsne'  # Fijo en t-SNE
    df_filtrado = df.copy()
    if curso_seleccionado != 'Todos':
        df_filtrado = df[df['curso'] == curso_seleccionado]
    # Si se selecciona 'numero+curso', crear columna auxiliar
    if color_by == 'numero+curso':
        df_filtrado = df_filtrado.copy()
        df_filtrado['numero+curso'] = df_filtrado['numero'].astype(str) + ' - ' + df_filtrado['curso']
        color_col = 'numero+curso'
    else:
        color_col = color_by
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Capítulos", len(df_filtrado), help="Número total de capítulos en la selección actual")
    with col2:
        st.metric("Cursos Únicos", df_filtrado['curso'].nunique(), help="Número de cursos diferentes")
    with col3:
        st.metric("Dimensión Embeddings", len(df.iloc[0]['embeddings_array']) if len(df) > 0 else 0, help="Dimensionalidad original de los embeddings")
    if len(df_filtrado) == 0:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return
    # Cargar reducción precomputada
    reduced_embeddings = cargar_dim_reduction(metodo)
    # Filtrar reducción si hay filtro de curso
    if curso_seleccionado != 'Todos':
        mask = df['curso'] == curso_seleccionado
        reduced_embeddings = reduced_embeddings[mask.values]
    # Crear gráfico
    plot_df = df_filtrado.copy()
    plot_df['x'] = reduced_embeddings[:, 0]
    plot_df['y'] = reduced_embeddings[:, 1]
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color=color_col,
        hover_data={
            'etiqueta': True,
            'keywords': True,
            'curso': True,
            'x': False,
            'y': False
        },
        title=f"Visualización de Embeddings por {color_col}",
        width=800,
        height=600
    )
    fig.update_layout(xaxis_title="Dimensión 1", yaxis_title="Dimensión 2", showlegend=True, hovermode='closest')
    fig.update_traces(textposition="middle right", textfont_size=8, marker=dict(size=10, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📋 Datos Detallados")
    st.dataframe(
        df_filtrado[['curso', 'numero', 'titulo', 'keywords', 'etiqueta']].rename(columns={
            'curso': 'Curso',
            'numero': 'Número',
            'titulo': 'Título',
            'keywords': 'Palabras Clave',
            'etiqueta': 'Etiqueta Completa'
        }),
        use_container_width=True,
        height=300
    )

# Esta parte eliminada del archivo porque estaba duplicada

def calculate_similarity_matrix(embeddings_matrix, method='cosine'):
    """Calcular matriz de similitud/distancia usando diferentes métodos"""
    if method == 'cosine':
        # Similitud coseno sin normalización adicional
        similarity_matrix = cosine_similarity(embeddings_matrix)
    elif method == 'euclidean':
        # Distancia euclidiana directa (valores más altos = más distantes)
        similarity_matrix = euclidean_distances(embeddings_matrix)
    elif method == 'dot_product':
        # Producto punto directo sin normalización
        similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
    
    return similarity_matrix

def analyze_intra_chapter_keywords(df):
    """Analizar métricas de dispersión/cohesión dentro de cada capítulo"""
    
    intra_metrics = []
    
    # Asegurar que estamos procesando cada capítulo una sola vez
    df_unique = df.drop_duplicates(subset=['id']).copy()
    
    for idx, row in df_unique.iterrows():
        keywords = row['keywords']
        if pd.notna(keywords) and str(keywords).strip():
            # Limpiar y separar keywords
            keywords_clean = str(keywords).replace('•', '').replace('-', '').strip()
            keywords_list = [k.strip() for k in keywords_clean.split('\n') if k.strip()]
            
            # Métricas básicas
            num_keywords = len(keywords_list)
            total_chars = sum(len(k) for k in keywords_list) if keywords_list else 0
            avg_keyword_length = total_chars / num_keywords if num_keywords > 0 else 0
            
            # Diversidad léxica (palabras únicas vs total)
            all_words = []
            for keyword in keywords_list:
                all_words.extend(keyword.lower().split())
            
            unique_words = len(set(all_words)) if all_words else 0
            total_words = len(all_words)
            lexical_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Longitud total del texto de keywords
            total_text_length = len(keywords_clean)
            
        else:
            # Valores por defecto para capítulos sin keywords
            num_keywords = 0
            avg_keyword_length = 0.0
            lexical_diversity = 0.0
            total_text_length = 0
            total_words = 0
            unique_words = 0
        
        intra_metrics.append({
            'id': row['id'],
            'curso': row['curso'],
            'numero': row['numero'],
            'titulo': row['titulo'],
            'etiqueta': row['etiqueta'],
            'num_keywords': num_keywords,
            'avg_keyword_length': avg_keyword_length,
            'lexical_diversity': lexical_diversity,
            'total_text_length': total_text_length,
            'total_words': total_words,
            'unique_words': unique_words
        })
    
    return pd.DataFrame(intra_metrics)

def calculate_intra_chapter_embedding_distances(df):
    """Calcular distancias internas usando embeddings"""
    
    dispersions = []
    
    # Asegurar que estamos procesando cada capítulo una sola vez
    df_unique = df.drop_duplicates(subset=['id']).copy()
    
    for idx, row in df_unique.iterrows():
        embedding = row['embeddings_array']
        if embedding is not None and len(embedding) > 0:
            try:
                # Magnitud del vector (como proxy de "densidad" conceptual)
                magnitude = float(np.linalg.norm(embedding))
                
                # Varianza de las dimensiones (como proxy de dispersión conceptual)
                variance = float(np.var(embedding))
                
                # Entropía normalizada de las dimensiones
                abs_embedding = np.abs(embedding)
                sum_abs = np.sum(abs_embedding)
                if sum_abs > 0:
                    normalized_embedding = abs_embedding / sum_abs
                    # Evitar log(0) agregando pequeño epsilon
                    entropy = float(-np.sum(normalized_embedding * np.log(normalized_embedding + 1e-10)))
                else:
                    entropy = 0.0
                
            except Exception as e:
                print(f"Error procesando embedding para capítulo {row['id']}: {e}")
                magnitude = 0.0
                variance = 0.0
                entropy = 0.0
        else:
            magnitude = 0.0
            variance = 0.0
            entropy = 0.0
        
        dispersions.append({
            'id': row['id'],
            'etiqueta': row['etiqueta'],
            'embedding_magnitude': magnitude,
            'embedding_variance': variance,
            'embedding_entropy': entropy
        })
    
    return pd.DataFrame(dispersions)

def create_complete_metrics_table(df_filtrado, similarity_matrix, combined_metrics, similarity_method):
    """Crear tabla completa con todas las métricas inter e intra-capítulo - SOLO 99 CAPÍTULOS"""
    
    # Asegurar que trabajamos solo con capítulos únicos
    df_unique = df_filtrado.drop_duplicates(subset=['id']).copy().reset_index(drop=True)
    
    # Verificar que tenemos exactamente la cantidad correcta de capítulos
    expected_chapters = len(df_unique)
    
    # Calcular promedio de similitud/distancia para cada capítulo
    avg_similarities = []
    for i in range(min(len(similarity_matrix), expected_chapters)):
        # Excluir la similitud consigo mismo (diagonal)
        similarities = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
        avg_similarities.append(np.mean(similarities))
    
    # Crear DataFrame base con EXACTAMENTE los capítulos únicos
    complete_table = pd.DataFrame({
        'Capítulo': df_unique['etiqueta'].tolist()[:len(avg_similarities)],
        'Curso': df_unique['curso'].tolist()[:len(avg_similarities)],
        'Número': df_unique['numero'].tolist()[:len(avg_similarities)],
        'Título': df_unique['titulo'].tolist()[:len(avg_similarities)],
        'Métrica_Promedio': avg_similarities
    })
    
    # Solo agregar métricas intra-capítulo si combined_metrics existe y tiene datos
    if combined_metrics is not None and len(combined_metrics) > 0:
        # Asegurar que combined_metrics también tenga capítulos únicos
        combined_unique = combined_metrics.drop_duplicates(subset=['id']).copy().reset_index(drop=True)
        
        # Agregar métricas intra-capítulo de forma segura
        metrics_to_add = {
            'num_keywords': 'Num_Keywords',
            'lexical_diversity': 'Diversidad_Léxica',
            'embedding_magnitude': 'Magnitud_Embedding', 
            'embedding_variance': 'Varianza_Embedding',
            'embedding_entropy': 'Entropía_Embedding'
        }
        
        for metric, new_name in metrics_to_add.items():
            if metric in combined_unique.columns:
                # Tomar solo los primeros valores hasta el tamaño de complete_table
                values_to_add = combined_unique[metric].values[:len(complete_table)]
                if len(values_to_add) == len(complete_table):
                    complete_table[new_name] = values_to_add
    
    # Redondear valores numéricos
    numeric_columns = ['Métrica_Promedio']
    if 'Diversidad_Léxica' in complete_table.columns:
        numeric_columns.append('Diversidad_Léxica')
    if 'Magnitud_Embedding' in complete_table.columns:
        numeric_columns.append('Magnitud_Embedding')
    if 'Varianza_Embedding' in complete_table.columns:
        numeric_columns.append('Varianza_Embedding')
    if 'Entropía_Embedding' in complete_table.columns:
        numeric_columns.append('Entropía_Embedding')
    
    for col in numeric_columns:
        complete_table[col] = complete_table[col].round(3)
    
    return complete_table

def create_similarity_heatmap(similarity_matrix, labels, title, method):
    """Crear heatmap de similitud"""
    
    # Configurar colorscale según el método
    if method == 'euclidean':
        colorscale = 'Viridis_r'  # Invertido para distancias (rojo = mayor distancia)
        colorbar_title = 'Distancia Euclidiana'
    elif method == 'cosine':
        colorscale = 'RdBu'
        colorbar_title = 'Similitud Coseno'
    else:  # dot_product
        colorscale = 'Blues'
        colorbar_title = 'Producto Punto'
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=colorbar_title),
        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Similitud: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Capítulos",
        yaxis_title="Capítulos",
        width=800,
        height=800,
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )
    
    return fig

def create_average_similarity_heatmap(similarity_matrix, labels, courses, title, method):
    """Crear heatmap del promedio de similitud por capítulo"""
    
    # Calcular promedio de similitud para cada capítulo (excluyendo la diagonal)
    avg_similarities = []
    for i in range(len(similarity_matrix)):
        # Excluir la similitud consigo mismo (diagonal)
        similarities = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
        avg_similarities.append(np.mean(similarities))
    
    # Crear DataFrame para mejor visualización
    df_avg = pd.DataFrame({
        'Capítulo': labels,
        'Curso': courses,
        'Similitud_Promedio': avg_similarities
    })
    
    # Ordenar por curso y número para mejor visualización
    df_avg = df_avg.sort_values(['Curso', 'Capítulo'])
    
    # Crear heatmap horizontal
    fig = go.Figure(data=go.Heatmap(
        z=[df_avg['Similitud_Promedio'].values],
        y=['Promedio de Similitud'],
        x=df_avg['Capítulo'].values,
        colorscale='Viridis_r' if method == 'euclidean' else ('RdBu' if method == 'cosine' else 'Blues'),
        showscale=True,
        colorbar=dict(title=f'Similitud Promedio ({method})'),
        hovertemplate='<b>%{x}</b><br>Similitud Promedio: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Capítulos",
        yaxis_title="",
        width=1000,
        height=200,
        xaxis=dict(tickangle=45, tickfont=dict(size=8))
    )
    
    return fig, df_avg

def similarity_heatmaps_tab():
    """Pestaña para heatmaps de similitud"""
    st.markdown('<h1 class="main-header">🔥 Matrices de Similitud</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el archivo CSV esté en la carpeta 'data/'.")
        return
    
    # --- MOVER SELECTORES AL CUERPO DE LA PESTAÑA ---
    st.subheader("Configuración de Similitud")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        similarity_method = st.selectbox(
        "Método de Similitud:",
        ['cosine', 'euclidean', 'dot_product'],
        format_func=lambda x: {
            'cosine': 'Similitud Coseno',
            'euclidean': 'Distancia Euclidiana',
            'dot_product': 'Producto Punto'
        }[x],
        help="""
        - **Coseno**: Similitud coseno sin normalización adicional (-1 a 1)
        - **Euclidiana**: Distancia euclidiana directa (valores más altos = más distantes)
        - **Producto Punto**: Producto punto directo sin normalización
        """,
        key="similarity_heatmaps_method"
    )
    cursos_disponibles = ['Todos'] + ordenar_cursos_personalizado(df['curso'].unique().tolist())
    with col_sel2:
        curso_seleccionado = st.selectbox(
        "Filtrar por Curso:",
        cursos_disponibles,
            help="Filtrar matriz por curso específico",
            key="sim_heatmaps_curso"
    )
    
    # Filtrar datos
    df_filtrado = df.copy()
    if curso_seleccionado != 'Todos':
        df_filtrado = df[df['curso'] == curso_seleccionado]
    
    if len(df_filtrado) == 0:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return
    
    # Asegurar que trabajamos con capítulos únicos
    df_unique = df_filtrado.drop_duplicates(subset=['id']).copy()
    
    # Crear identificador único de capítulo para todo el análisis de similitud
    def capitulo_id(row):
        # Ejemplo: '1B: Capítulo N°3: Suma y resta'
        curso_abbr = str(row['curso']).replace('Primero', '1B').replace('Segundo', '2B').replace('Tercero', '3B').replace('Cuarto', '4B').replace('Quinto', '5B').replace('Sexto', '6B')
        return f"{curso_abbr}: Capítulo N°{row['numero']}: {row['titulo']}"
    
    df_unique['capitulo_id'] = df_unique.apply(capitulo_id, axis=1)
    labels = df_unique['capitulo_id'].tolist()

    # Preparar datos usando capítulos únicos
    embeddings_matrix = np.vstack(df_unique['embeddings_array'].values)
    courses = df_unique['curso'].tolist()

    # Mostrar información
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Capítulos Únicos Analizados", len(df_unique))
    with col2:
        st.metric("Dimensión Embeddings", embeddings_matrix.shape[1])
    with col3:
        st.metric("Método Seleccionado", {
            'cosine': 'Coseno',
            'euclidean': 'Euclidiana', 
            'dot_product': 'Producto Punto'
        }[similarity_method])
    
    # Información adicional sobre los datos
    if len(df_filtrado) != len(df_unique):
        st.info(f"ℹ️ Se detectaron {len(df_filtrado)} filas totales, usando {len(df_unique)} capítulos únicos para el análisis.")
    
    # Calcular matriz de similitud
    with st.spinner(f"Calculando matriz de similitud usando {similarity_method}..."):
        similarity_matrix = calculate_similarity_matrix(embeddings_matrix, similarity_method)
    
    # --- RESTO DEL CÓDIGO IGUAL ---
    # Mostrar estadísticas de la matriz
    st.subheader("📊 Estadísticas de la Matriz de Similitud")
    
    # Excluir diagonal para estadísticas
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    if similarity_method == 'euclidean':
        # Para distancia euclidiana, valores más altos = más distantes
        with col1:
            st.metric("Distancia Promedio", f"{np.mean(upper_triangle):.3f}")
        with col2:
            st.metric("Distancia Máxima", f"{np.max(upper_triangle):.3f}")
        with col3:
            st.metric("Distancia Mínima", f"{np.min(upper_triangle):.3f}")
        with col4:
            st.metric("Desviación Estándar", f"{np.std(upper_triangle):.3f}")
    else:
        # Para coseno y producto punto, valores más altos = más similares
        with col1:
            st.metric("Similitud Promedio", f"{np.mean(upper_triangle):.3f}")
        with col2:
            st.metric("Similitud Máxima", f"{np.max(upper_triangle):.3f}")
        with col3:
            st.metric("Similitud Mínima", f"{np.min(upper_triangle):.3f}")
        with col4:
            st.metric("Desviación Estándar", f"{np.std(upper_triangle):.3f}")
    
    # Matriz de similitud completa
    st.subheader("🔥 Matriz de Similitud Inter-Capítulo")
    
    method_names = {
        'cosine': 'Similitud Coseno',
        'euclidean': 'Similitud Euclidiana',
        'dot_product': 'Producto Punto'
    }
    
    try:
        # Debug: verificar datos antes de crear el gráfico
        st.write(f"Debug: similarity_matrix shape: {similarity_matrix.shape}")
        st.write(f"Debug: labels length: {len(labels)}")
        st.write(f"Debug: method: {similarity_method}")
        
        fig_matrix = create_similarity_heatmap(
            similarity_matrix, 
            labels, 
            f"Matriz de Similitud Inter-Capítulo ({method_names[similarity_method]})",
            similarity_method
        )
        
        if fig_matrix is not None:
            st.plotly_chart(fig_matrix, use_container_width=True)
        else:
            st.error("Error: create_similarity_heatmap devolvió None")
            
    except Exception as e:
        st.error(f"Error creando matriz de similitud: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Heatmap de similitud promedio
    st.subheader("📈 Similitud Promedio por Capítulo")
    
    try:
        # Calcular promedio de similitud para cada capítulo (excluyendo la diagonal)
        avg_similarities = []
        for i in range(len(similarity_matrix)):
            # Excluir la similitud consigo mismo (diagonal)
            similarities = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
            avg_similarities.append(np.mean(similarities))
        
        # Crear DataFrame para mejor visualización
        df_avg = pd.DataFrame({
            'Capítulo': labels,
            'Curso': courses,
            'Similitud_Promedio': avg_similarities
        })
        
        # Ordenar por curso y número para mejor visualización
        df_avg = df_avg.sort_values(['Curso', 'Capítulo'])
        
        # Gráfico de barras horizontal
        fig_avg = px.bar(
            df_avg,
            y='Capítulo',
            x='Similitud_Promedio',
            color='Curso',
            orientation='h',
            title=f"Similitud Promedio por Capítulo ({method_names[similarity_method]})",
            labels={
                'Similitud_Promedio': f'{"Distancia" if similarity_method == "euclidean" else "Similitud"} Promedio',
                'Capítulo': 'Capítulos'
            },
            height=800
        )
        fig_avg.update_layout(
            yaxis=dict(tickfont=dict(size=8)),
            showlegend=True,
            xaxis_title=f'{"Distancia" if similarity_method == "euclidean" else "Similitud"} Promedio',
            yaxis_title="Capítulos"
        )
        st.plotly_chart(fig_avg, use_container_width=True)
        st.write(f"Debug: Se creó gráfico con {len(df_avg)} capítulos")
        
    except Exception as e:
        st.error(f"Error creando gráfico promedio: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Análisis intra-capítulo
    st.subheader("🔍 Análisis Intra-Capítulo: Dispersión de Keywords")
    
    # Calcular métricas intra-capítulo
    with st.spinner("Calculando métricas de dispersión interna..."):
        try:
            intra_metrics = analyze_intra_chapter_keywords(df_unique)
            embedding_dispersions = calculate_intra_chapter_embedding_distances(df_unique)
            
            # Verificar que ambos DataFrames tengan el mismo número de filas
            if len(intra_metrics) != len(embedding_dispersions):
                st.error(f"Error: intra_metrics tiene {len(intra_metrics)} filas, pero embedding_dispersions tiene {len(embedding_dispersions)} filas")
                return
            
            # Combinar métricas de forma más segura
            if len(intra_metrics) > 0 and len(embedding_dispersions) > 0:
                # Asegurar que los IDs coincidan
                intra_metrics = intra_metrics.sort_values('id')
                embedding_dispersions = embedding_dispersions.sort_values('id')
                
                combined_metrics = pd.merge(intra_metrics, embedding_dispersions, on=['id', 'etiqueta'], how='inner')
                
                # Verificar que el merge funcionó correctamente
                if len(combined_metrics) != len(df_unique):
                    st.warning(f"Atención: combined_metrics tiene {len(combined_metrics)} filas, pero se esperaban {len(df_unique)}")
            else:
                st.error("Error: No se pudieron calcular las métricas intra-capítulo")
                return
                
        except Exception as e:
            st.error(f"Error calculando métricas intra-capítulo: {str(e)}")
            return
    
    # Mostrar métricas estadísticas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Promedio Keywords/Capítulo", f"{combined_metrics['num_keywords'].mean():.1f}")
    with col2:
        st.metric("Diversidad Léxica Promedio", f"{combined_metrics['lexical_diversity'].mean():.3f}")
    with col3:
        st.metric("Magnitud Embedding Promedio", f"{combined_metrics['embedding_magnitude'].mean():.1f}")
    with col4:
        st.metric("Varianza Embedding Promedio", f"{combined_metrics['embedding_variance'].mean():.3f}")
    
    # Visualizaciones de dispersión interna
    tab_intra1, tab_intra2, tab_intra3 = st.tabs(["📝 Métricas Textuales", "🧮 Métricas de Embeddings", "📊 Correlaciones"])
    
    with tab_intra1:
        # Gráficos de métricas textuales
        col1, col2 = st.columns(2)
        
        with col1:
            fig_keywords = px.scatter(
                combined_metrics,
                x='num_keywords',
                y='lexical_diversity',
                color='curso',
                hover_data=['etiqueta'],
                title="Número de Keywords vs Diversidad Léxica",
                labels={'num_keywords': 'Número de Keywords', 'lexical_diversity': 'Diversidad Léxica'}
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
        
        with col2:
            fig_length = px.box(
                combined_metrics,
                x='curso',
                y='avg_keyword_length',
                title="Distribución de Longitud Promedio de Keywords por Curso",
                labels={'avg_keyword_length': 'Longitud Promedio Keywords'}
            )
            fig_length.update_layout(xaxis=dict(tickangle=45))
            st.plotly_chart(fig_length, use_container_width=True)
    
    with tab_intra2:
        # Gráficos de métricas de embeddings
        col1, col2 = st.columns(2)
        
        with col1:
            fig_magnitude = px.scatter(
                combined_metrics,
                x='embedding_magnitude',
                y='embedding_variance',
                color='curso',
                hover_data=['etiqueta'],
                title="Magnitud vs Varianza de Embeddings",
                labels={'embedding_magnitude': 'Magnitud del Embedding', 'embedding_variance': 'Varianza del Embedding'}
            )
            st.plotly_chart(fig_magnitude, use_container_width=True)
        
        with col2:
            fig_entropy = px.histogram(
                combined_metrics,
                x='embedding_entropy',
                color='curso',
                title="Distribución de Entropía de Embeddings",
                labels={'embedding_entropy': 'Entropía del Embedding'},
                marginal="box"
            )
            st.plotly_chart(fig_entropy, use_container_width=True)
    
    with tab_intra3:
        # Matriz de correlación
        numeric_cols = ['num_keywords', 'avg_keyword_length', 'lexical_diversity', 
                       'total_text_length', 'embedding_magnitude', 'embedding_variance', 'embedding_entropy']
        
        correlation_matrix = combined_metrics[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title='Correlación'),
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlación: %{z:.3f}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            title="Matriz de Correlación - Métricas Intra-Capítulo",
            xaxis_title="Métricas",
            yaxis_title="Métricas",
            width=600,
            height=600,
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tabla de métricas por capítulo
        st.write("**Métricas Detalladas por Capítulo:**")
        
        # Verificar que tenemos los datos correctos
        if len(combined_metrics) == len(df_unique):
            display_cols = ['etiqueta', 'curso', 'num_keywords', 'lexical_diversity', 
                           'embedding_magnitude', 'embedding_variance', 'embedding_entropy']
            
            # Verificar que todas las columnas existen
            available_cols = [col for col in display_cols if col in combined_metrics.columns]
            
            if available_cols:
                display_metrics = combined_metrics[available_cols].copy()
                
                # Redondear valores numéricos
                numeric_cols = ['lexical_diversity', 'embedding_magnitude', 'embedding_variance', 'embedding_entropy']
                for col in numeric_cols:
                    if col in display_metrics.columns:
                        display_metrics[col] = display_metrics[col].round(3)
                
                # Renombrar columnas para mejor legibilidad
                column_names = {
                    'etiqueta': 'Capítulo',
                    'curso': 'Curso', 
                    'num_keywords': 'N° Keywords',
                    'lexical_diversity': 'Diversidad Léxica',
                    'embedding_magnitude': 'Magnitud Embedding',
                    'embedding_variance': 'Varianza Embedding',
                    'embedding_entropy': 'Entropía Embedding'
                }
                
                display_metrics = display_metrics.rename(columns=column_names)
                st.dataframe(display_metrics, use_container_width=True, height=300)
            else:
                st.error("No se encontraron las columnas esperadas en combined_metrics")
        else:
            st.error(f"Error: combined_metrics tiene {len(combined_metrics)} filas, pero df_unique tiene {len(df_unique)} filas")
    
    # Tabla de rankings
    st.subheader("🏆 Ranking de Similitud Promedio")
    
    col1, col2 = st.columns(2)
    
    if similarity_method == 'euclidean':
        # Para distancia euclidiana: menor distancia = más similar
        with col1:
            st.write("**Top 10 - Menor Distancia Promedio (Más Similares):**")
            top_similar = df_avg.nsmallest(10, 'Similitud_Promedio')[['Capítulo', 'Curso', 'Similitud_Promedio']].copy()
            top_similar = top_similar.rename(columns={'Similitud_Promedio': 'Distancia_Promedio'})
            top_similar['Distancia_Promedio'] = top_similar['Distancia_Promedio'].round(3)
            st.dataframe(top_similar, hide_index=True)
        
        with col2:
            st.write("**Top 10 - Mayor Distancia Promedio (Más Únicos):**")
            bottom_similar = df_avg.nlargest(10, 'Similitud_Promedio')[['Capítulo', 'Curso', 'Similitud_Promedio']].copy()
            bottom_similar = bottom_similar.rename(columns={'Similitud_Promedio': 'Distancia_Promedio'})
            bottom_similar['Distancia_Promedio'] = bottom_similar['Distancia_Promedio'].round(3)
            st.dataframe(bottom_similar, hide_index=True)
    else:
        # Para coseno y producto punto: mayor valor = más similar
        with col1:
            st.write("**Top 10 - Mayor Similitud Promedio:**")
            top_similar = df_avg.nlargest(10, 'Similitud_Promedio')[['Capítulo', 'Curso', 'Similitud_Promedio']].copy()
            top_similar['Similitud_Promedio'] = top_similar['Similitud_Promedio'].round(3)
            st.dataframe(top_similar, hide_index=True)
        
        with col2:
            st.write("**Top 10 - Menor Similitud Promedio (Más Únicos):**")
            bottom_similar = df_avg.nsmallest(10, 'Similitud_Promedio')[['Capítulo', 'Curso', 'Similitud_Promedio']].copy()
            bottom_similar['Similitud_Promedio'] = bottom_similar['Similitud_Promedio'].round(3)
            st.dataframe(bottom_similar, hide_index=True)
    
    # Análisis de pares más similares
    st.subheader("🎯 Pares de Capítulos Más Similares")
    
    # Encontrar los pares más similares
    pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            pairs.append((i, j, similarity_matrix[i, j], labels[i], labels[j], courses[i], courses[j]))
    
    # Ordenar según la métrica
    if similarity_method == 'euclidean':
        # Para distancia: menor valor = más similar
        pairs.sort(key=lambda x: x[2], reverse=False)
        st.write("**Top 10 Pares con Menor Distancia (Más Similares):**")
        pairs_df = pd.DataFrame(pairs[:10], columns=[
            'idx1', 'idx2', 'Distancia', 'Capítulo_1', 'Capítulo_2', 'Curso_1', 'Curso_2'
        ])[['Capítulo_1', 'Curso_1', 'Capítulo_2', 'Curso_2', 'Distancia']]
        pairs_df['Distancia'] = pairs_df['Distancia'].round(3)
        st.dataframe(pairs_df, hide_index=True, use_container_width=True)
    else:
        # Para similitud: mayor valor = más similar
        pairs.sort(key=lambda x: x[2], reverse=True)
        st.write("**Top 10 Pares Más Similares:**")
        pairs_df = pd.DataFrame(pairs[:10], columns=[
            'idx1', 'idx2', 'Similitud', 'Capítulo_1', 'Capítulo_2', 'Curso_1', 'Curso_2'
        ])[['Capítulo_1', 'Curso_1', 'Capítulo_2', 'Curso_2', 'Similitud']]
        pairs_df['Similitud'] = pairs_df['Similitud'].round(3)
        st.dataframe(pairs_df, hide_index=True, use_container_width=True)
    
    # Tabla completa navegable de todos los capítulos
    st.subheader("📋 Tabla Completa de Métricas Inter-Capítulo")
    st.write("""
    **Lista navegable y filtrable de los 99 capítulos con métricas calculadas a nivel de capítulo:**
    
    - **Distancia Promedio Inter-Capítulo**: Promedio de distancia/similitud de cada capítulo con todos los demás capítulos
    - **Métricas Intra-Capítulo**: Características internas de las keywords de cada capítulo
    - **Todas las comparaciones son entre los 99 capítulos**, no entre keywords individuales
    """)
    
    # Crear tabla completa con todas las métricas
    try:
        if 'combined_metrics' in locals() and len(combined_metrics) > 0:
            complete_metrics = create_complete_metrics_table(df_unique, similarity_matrix, combined_metrics, similarity_method)
        else:
            st.error("No se pueden crear las métricas completas: combined_metrics no está disponible")
            return
    except Exception as e:
        st.error(f"Error creando tabla completa: {str(e)}")
        return
    
    # Filtros para la tabla
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        # Filtro por curso en la tabla
        cursos_tabla = ['Todos'] + ordenar_cursos_personalizado(complete_metrics['Curso'].unique().tolist())
        curso_tabla_filtro = st.selectbox(
            "Filtrar Tabla por Curso:",
            cursos_tabla,
            key="curso_tabla_filter"
        )
    
    with col_filter2:
        # Filtro por rango de similitud/distancia
        if similarity_method == 'euclidean':
            min_val, max_val = st.slider(
                "Rango de Distancia:",
                0.0,
                2.0,
                (0.0, 2.0),
                step=0.01,
                key="distance_range"
            )
        else:
            min_val, max_val = st.slider(
                "Rango de Similitud:",
                0.0,
                2.0,
                (0.0, 2.0),
                step=0.01,
                key="similarity_range"
            )
    
    with col_filter3:
        # Ordenar por
        # Verificar qué columnas están disponibles  
        available_columns = complete_metrics.columns.tolist()
        
        sort_options = {
            'Métrica_Promedio': f'{"Distancia" if similarity_method == "euclidean" else "Similitud"} Promedio',
            'Número': 'Número de Capítulo'
        }
        
        # Agregar opciones solo si las columnas existen
        if 'Num_Keywords' in available_columns:
            sort_options['Num_Keywords'] = 'Número de Keywords'
        if 'Diversidad_Léxica' in available_columns:
            sort_options['Diversidad_Léxica'] = 'Diversidad Léxica'
        if 'Magnitud_Embedding' in available_columns:
            sort_options['Magnitud_Embedding'] = 'Magnitud Embedding'
        
        sort_by = st.selectbox(
            "Ordenar por:",
            list(sort_options.keys()),
            format_func=lambda x: sort_options[x],
            key="sort_by_table"
        )
        
        ascending = st.checkbox("Orden Ascendente", value=similarity_method=='euclidean', key="ascending_order")
    
    # Aplicar filtros
    filtered_table = complete_metrics.copy()
    
    if curso_tabla_filtro != 'Todos':
        filtered_table = filtered_table[filtered_table['Curso'] == curso_tabla_filtro]
    
    filtered_table = filtered_table[
        (filtered_table['Métrica_Promedio'] >= min_val) & 
        (filtered_table['Métrica_Promedio'] <= max_val)
    ]
    
    # Ordenar
    filtered_table = filtered_table.sort_values(sort_by, ascending=ascending)
    
    # Agregar ranking
    filtered_table = filtered_table.reset_index(drop=True)
    filtered_table.insert(0, 'Ranking', range(1, len(filtered_table) + 1))
    
    # Mostrar estadísticas de la tabla filtrada
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("Capítulos Mostrados", len(filtered_table))
    with col_stats2:
        st.metric("Cursos Únicos", filtered_table['Curso'].nunique())
    with col_stats3:
        if similarity_method == 'euclidean':
            st.metric("Distancia Promedio", f"{filtered_table['Métrica_Promedio'].mean():.3f}")
        else:
            st.metric("Similitud Promedio", f"{filtered_table['Métrica_Promedio'].mean():.3f}")
    with col_stats4:
        if 'Num_Keywords' in filtered_table.columns:
            st.metric("Keywords Promedio", f"{filtered_table['Num_Keywords'].mean():.1f}")
        else:
            st.metric("Capítulos con Datos", len(filtered_table[filtered_table['Métrica_Promedio'] > 0]))
    
    # Configurar columnas dinámicamente
    column_config = {
        "Ranking": st.column_config.NumberColumn("Rank", width="small"),
        "Capítulo": st.column_config.TextColumn("Capítulo", width="large"),
        "Curso": st.column_config.TextColumn("Curso", width="medium"),
        "Número": st.column_config.NumberColumn("N°", width="small"),
        "Métrica_Promedio": st.column_config.NumberColumn(
            f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'} Prom.",
            format="%.3f",
            width="medium"
        )
    }
    
    # Agregar configuraciones solo para columnas que existen
    if "Num_Keywords" in filtered_table.columns:
        column_config["Num_Keywords"] = st.column_config.NumberColumn("Keywords", width="small")
    if "Diversidad_Léxica" in filtered_table.columns:
        column_config["Diversidad_Léxica"] = st.column_config.NumberColumn("Div. Léxica", format="%.3f", width="medium")
    if "Magnitud_Embedding" in filtered_table.columns:
        column_config["Magnitud_Embedding"] = st.column_config.NumberColumn("Magnitud", format="%.1f", width="medium")
    if "Varianza_Embedding" in filtered_table.columns:
        column_config["Varianza_Embedding"] = st.column_config.NumberColumn("Varianza", format="%.3f", width="medium")
    
    # Mostrar tabla con scroll
    st.dataframe(
        filtered_table,
        use_container_width=True,
        height=600,
        column_config=column_config
    )
    
    # Resumen estadístico por curso
    if len(filtered_table) > 0:
        st.subheader("📊 Resumen por Curso")
        
        # Crear diccionario de agregación dinámicamente
        agg_dict = {
            'Métrica_Promedio': ['mean', 'std', 'min', 'max']
        }
        
        # Agregar métricas solo si las columnas existen
        if 'Num_Keywords' in filtered_table.columns:
            agg_dict['Num_Keywords'] = 'mean'
        if 'Diversidad_Léxica' in filtered_table.columns:
            agg_dict['Diversidad_Léxica'] = 'mean'
        if 'Magnitud_Embedding' in filtered_table.columns:
            agg_dict['Magnitud_Embedding'] = 'mean'
        
        summary_by_course = filtered_table.groupby('Curso').agg(agg_dict).round(3)
        
        # Aplanar nombres de columnas
        summary_by_course.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in summary_by_course.columns]
        summary_by_course = summary_by_course.reset_index()
        
        st.dataframe(summary_by_course, use_container_width=True)
    
    # Descargar datos
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        if st.button("💾 Descargar Matriz de Similitud como CSV"):
            # Crear DataFrame de la matriz
            matrix_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
            csv = matrix_df.to_csv()
            file_name_matrix = f"matriz_similitud_{similarity_method}_{safe_filename(curso_seleccionado)}.csv"
            debug_string("file_name_matrix", file_name_matrix)
            st.download_button(
                label="Descargar Matriz CSV",
                data=csv,
                file_name=file_name_matrix,
                mime="text/csv",
                key="download_matrix"
            )
    
    with col_download2:
        if st.button("📊 Descargar Tabla de Métricas como CSV"):
            csv_metrics = filtered_table.to_csv(index=False)
            file_name_metrics = f"metricas_completas_{similarity_method}_{safe_filename(curso_tabla_filtro)}.csv"
            debug_string("file_name_metrics", file_name_metrics)
            st.download_button(
                label="Descargar Métricas CSV",
                data=csv_metrics,
                file_name=file_name_metrics,
                mime="text/csv",
                key="download_metrics"
            )

def debug_string(label, value):
    try:
        st.write(f"DEBUG {label}: {repr(value)}")
        print(f"DEBUG {label}: {repr(value)}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG ERROR printing {label}: {e}", file=sys.stderr)

def buscar_con_faiss(embedding_pregunta: np.ndarray, index, metadata, method: str = 'cosine', k: int = None, umbral_distancia: float = 0.0):
    """
    Busca keywords similares usando FAISS y luego agrupa por capítulo.
    """
    embedding_query = embedding_pregunta.astype('float32').reshape(1, -1)
    k_busqueda = index.ntotal if k is None else k
    scores, indices = index.search(embedding_query, k_busqueda)
    keywords_resultados = []
    for score, idx in zip(scores[0], indices[0]):
        valor_faiss = float(score)
        if method == 'cosine':
            distancia_final = 1.0 - valor_faiss
        else:
            distancia_final = valor_faiss
        es_relevante = distancia_final <= umbral_distancia
        keywords_resultados.append({
            "id": metadata["ids"][idx],
            "curso": metadata["cursos"][idx],
            "numero": metadata["numeros"][idx],
            "titulo": metadata["titulos"][idx],
            "etiqueta": metadata["etiquetas"][idx],
            "keywords": metadata["keywords"][idx],
            "distancia": distancia_final,
            "es_relevante": es_relevante,
            "keyword_text": metadata["keywords"][idx]
        })
    capitulos_agrupados = {}
    for keyword_result in keywords_resultados:
        capitulo_id = keyword_result["id"]
        if capitulo_id not in capitulos_agrupados:
            capitulos_agrupados[capitulo_id] = {
                "id": keyword_result["id"],
                "curso": keyword_result["curso"],
                "numero": keyword_result["numero"],
                "titulo": keyword_result["titulo"],
                "etiqueta": keyword_result["etiqueta"],
                "keywords": keyword_result["keywords"],
                "distancia": keyword_result["distancia"],
                "es_relevante": keyword_result["es_relevante"],
                "keywords_relevantes": 1 if keyword_result["es_relevante"] else 0,
                "total_keywords": 1,
                "mejor_keyword": keyword_result["keyword_text"],
                "mejor_distancia": keyword_result["distancia"]
            }
        else:
            capitulo_actual = capitulos_agrupados[capitulo_id]
            if keyword_result["distancia"] < capitulo_actual["distancia"]:
                capitulo_actual["distancia"] = keyword_result["distancia"]
            if keyword_result["es_relevante"]:
                capitulo_actual["es_relevante"] = True
                capitulo_actual["keywords_relevantes"] += 1
            capitulo_actual["total_keywords"] += 1
            if keyword_result["distancia"] < capitulo_actual["mejor_distancia"]:
                capitulo_actual["mejor_keyword"] = keyword_result["keyword_text"]
                capitulo_actual["mejor_distancia"] = keyword_result["distancia"]
    resultados_capitulos = list(capitulos_agrupados.values())
    resultados_capitulos.sort(key=lambda x: x["distancia"])
    import pandas as pd
    return pd.DataFrame(resultados_capitulos)

# --- Búsqueda Semántica Tab ---
def semantic_search_tab():
    st.markdown('<h1 class="main-header">🔍 Búsqueda Semántica de Capítulos</h1>', unsafe_allow_html=True)
    # Cargar datos
    df = load_data()
    if df is None:
        return
    # Inicializar cliente OpenAI
    client = get_openai_client()
    if client is None:
        st.warning("⚠️ Para usar esta funcionalidad necesitas configurar tu API key de OpenAI en un archivo .env")
        st.code("OPENAI_API_KEY=tu_api_key_aqui", language="bash")
        return
    # Configuración dentro de la pestaña
    st.subheader("Configuración de búsqueda semántica")
    similarity_method = st.selectbox(
        "Método de Similitud:",
        ['cosine', 'euclidean', 'dot_product'],
        index=1,  # Euclidiana por defecto
        format_func=lambda x: {
            'cosine': 'Similitud Coseno',
            'euclidean': 'Distancia Euclidiana',
            'dot_product': 'Producto Punto'
        }[x],
        help="""
        - **Coseno**: Producto punto sin normalización (ángulo entre vectores)
        - **Euclidiana**: Distancia euclidiana directa (menor = más similar)
        - **Producto Punto**: Producto punto directo sin normalización
        """,
        key="semantic_search_method"
    )
    st.session_state.similarity_method = similarity_method
    st.markdown("### 💬 Ingresa tu búsqueda")
    st.info("Describe lo que estás buscando y encontraremos los capítulos más relevantes según sus contenidos.")
    # Input para la búsqueda
    prompt = st.text_area(
        "Escribe tu búsqueda:",
        placeholder="Ejemplo: ¿Cómo enseñar fracciones a estudiantes de cuarto básico?",
        height=100,
        key="semantic_search_prompt"
    )
    # Estado para guardar el embedding y los resultados
    if 'semantic_embedding' not in st.session_state:
        st.session_state.semantic_embedding = None
    if 'semantic_results' not in st.session_state:
        st.session_state.semantic_results = None
    if 'semantic_prompt' not in st.session_state:
        st.session_state.semantic_prompt = ''
    if 'semantic_umbral' not in st.session_state:
        st.session_state.semantic_umbral = 0.4
    if 'semantic_method' not in st.session_state:
        st.session_state.semantic_method = similarity_method
    
    # Botón para calcular embedding y similitud
    if st.button("🔍 Buscar", type="primary") and prompt.strip():
        t0 = time.time()
        with st.spinner("Generando embedding y calculando distancias..."):
            t1 = time.time()
            embedding_pregunta = obtener_embedding_pregunta(prompt, client)
            t2 = time.time()
            if embedding_pregunta is not None:
                # Cargar índice FAISS y metadatos
                index, metadata = cargar_indice_faiss(similarity_method)
                t3 = time.time()
                df_distancias = buscar_con_faiss(embedding_pregunta, index, metadata, method=similarity_method, k=None, umbral_distancia=0.4)
                t4 = time.time()
                st.session_state.semantic_embedding = embedding_pregunta
                st.session_state.semantic_results = df_distancias
                st.session_state.semantic_prompt = prompt
                st.session_state.semantic_umbral = 0.4
                st.session_state.semantic_method = similarity_method
                # Guardar métricas de tiempo y mejor resultado
                st.session_state.semantic_tiempo_embedding = t2 - t1
                st.session_state.semantic_tiempo_faiss = t4 - t3
                st.session_state.semantic_tiempo_total = t4 - t0
                if len(df_distancias) > 0:
                    mejor = df_distancias.iloc[0]
                    st.session_state.semantic_mejor = {
                        'capitulo': mejor['etiqueta'] if 'etiqueta' in mejor else '',
                        'curso': mejor['curso'] if 'curso' in mejor else '',
                        'numero': mejor['numero'] if 'numero' in mejor else '',
                        'mejor_keyword': mejor['mejor_keyword'] if 'mejor_keyword' in mejor else '',
                        'distancia': mejor['distancia'] if 'distancia' in mejor else None
                    }
                else:
                    st.session_state.semantic_mejor = None
    
    # Si ya hay embedding y resultados calculados, mostrar opciones y resultados
    if st.session_state.semantic_embedding is not None and st.session_state.semantic_results is not None:
        df_distancias = st.session_state.semantic_results
        similarity_method = st.session_state.semantic_method

        # Opciones de visualización (deben estar antes de usarlas)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            umbral_distancia = st.slider(
                "Umbral de distancia (máximo para resaltar)", 
                min_value=0.0, 
                max_value=2.0, 
                value=st.session_state.semantic_umbral, 
                step=0.01,
                help="Capítulos con al menos 1 keyword que tenga distancia ≤ este umbral se resaltarán. 0 = más similar para todos los métodos."
            )
        with col2:
            mostrar_todos = st.checkbox(
                "Mostrar todos los capítulos", 
                value=True,
                help="Si está desmarcado, solo muestra capítulos que superan el umbral"
            )
        with col3:
            mostrar_cant_keywords = st.checkbox(
                "Mostrar cantidad de keywords relevantes", 
                value=True,
                help="Muestra cuántas keywords de cada capítulo fueron relevantes respecto al umbral"
            )
            mostrar_mejor_keyword = st.checkbox(
                "Mostrar keyword más similar", 
                value=True,
                help="Muestra la keyword más similar de cada capítulo respecto a la búsqueda"
            )

        # Recalcular relevancia según el nuevo umbral
        df_distancias['es_relevante'] = df_distancias['distancia'] <= umbral_distancia
        df_distancias['keywords_relevantes'] = df_distancias['keywords_relevantes'] if 'keywords_relevantes' in df_distancias else 0
        df_distancias['total_keywords'] = df_distancias['total_keywords'] if 'total_keywords' in df_distancias else 0

        # --- NUEVO: Lista de capítulos con hit y total de keywords con hit ---
        # Para esto necesitamos la metadata original de keywords por capítulo y las distancias por keyword
        # Como buscar_con_faiss solo retorna la mejor keyword y el conteo, vamos a mostrar la suma de keywords_relevantes y la cantidad de capítulos con hit
        capitulos_con_hit = df_distancias[df_distancias['es_relevante']]
        cantidad_capitulos_hit = len(capitulos_con_hit)
        cantidad_keywords_hit = capitulos_con_hit['keywords_relevantes'].sum() if 'keywords_relevantes' in capitulos_con_hit else 0
        st.info(f"🔎 Capítulos con hit: {cantidad_capitulos_hit}\n🔑 Keywords con hit: {cantidad_keywords_hit}")
        if cantidad_capitulos_hit > 0:
            st.markdown("**Lista de capítulos con hit:**")
            for _, row in capitulos_con_hit.iterrows():
                st.markdown(f"- {row['etiqueta']} (Curso: {row['curso']}, N°: {row['numero']}) — Keywords relevantes: {row['keywords_relevantes']}")

        # Mostrar métricas de cómputo y mejor resultado
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("⏱️ Tiempo embedding", f"{st.session_state.get('semantic_tiempo_embedding', 0):.2f} s", help="Tiempo de cálculo del embedding de la pregunta")
        with colB:
            st.metric("⏱️ Tiempo FAISS", f"{st.session_state.get('semantic_tiempo_faiss', 0):.2f} s", help="Tiempo de búsqueda en el índice FAISS")
        with colC:
            st.metric("⏱️ Tiempo total", f"{st.session_state.get('semantic_tiempo_total', 0):.2f} s", help="Tiempo total de la consulta semántica")
        if st.session_state.get('semantic_mejor'):
            mejor = st.session_state['semantic_mejor']
            st.success(f"**Mejor capítulo:** {mejor['capitulo']} (Curso: {mejor['curso']}, N°: {mejor['numero']})\n\n**Mejor keyword:** {mejor['mejor_keyword']}\n\n**{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}:** {mejor['distancia']:.3f}")

        # --- NUEVO: Top 10 capítulos y keywords que hicieron hit ---
        st.markdown("### 🏅 Top 10 capítulos con keywords relevantes")
        top_10_hit = capitulos_con_hit.head(10)
        if len(top_10_hit) > 0:
            for i, (_, row) in enumerate(top_10_hit.iterrows(), 1):
                st.markdown(f"**{i}. {row['etiqueta']} (Curso: {row['curso']}, N°: {row['numero']})** — Keywords relevantes: {row['keywords_relevantes']}")
                # Mostrar todas las keywords que hicieron hit (si la metadata lo permite)
                # Si la función buscar_con_faiss no retorna las keywords individuales, solo mostramos la mejor
                if 'mejor_keyword' in row and row['mejor_keyword']:
                    st.markdown(f"- Mejor keyword: `{row['mejor_keyword']}` (distancia: {row['distancia']:.3f})")
                # Si en el futuro se retorna la lista de keywords que hicieron hit, aquí se mostrarían todas

        # Mostrar resultados (copiar la lógica de visualización actual)
        st.markdown("### 📚 Capítulos por Curso")
        st.markdown("*Los capítulos resaltados tienen al menos UNA keyword con distancia ≤ umbral*")
        cursos = ordenar_cursos_personalizado(df_distancias['curso'].unique().tolist())
        
        num_cols = min(len(cursos), 6)
        if num_cols > 0:
            cols = st.columns(num_cols)
            for i, curso in enumerate(cursos[:6]):
                with cols[i]:
                    st.markdown(f"#### {curso}")
                    capitulos_curso = df_distancias[df_distancias['curso'] == curso].copy()
                    capitulos_curso = capitulos_curso.sort_values('numero')
                    for _, capitulo in capitulos_curso.iterrows():
                        distancia = capitulo['distancia']
                        es_relevante = capitulo['es_relevante']
                        info_keywords = ""
                        if mostrar_cant_keywords and 'keywords_relevantes' in capitulo and 'total_keywords' in capitulo:
                            info_keywords = f" ({capitulo['keywords_relevantes']}/{capitulo['total_keywords']} keywords relevantes)"
                        info_mejor_keyword = ""
                        if mostrar_mejor_keyword and 'mejor_keyword' in capitulo:
                            # Sanitizar textos para HTML seguro
                            titulo_html = remove_accents_html(capitulo['titulo'])
                            mejor_keyword_html = remove_accents_html(capitulo['mejor_keyword']) if 'mejor_keyword' in capitulo else ''
                            info_mejor_keyword = f"<br><em>Keyword mas similar:</em> <strong>{mejor_keyword_html}</strong>"
                        if mostrar_todos or es_relevante:
                            if es_relevante:
                                if 'keywords_relevantes' in capitulo and 'total_keywords' in capitulo:
                                    proportion = capitulo['keywords_relevantes'] / capitulo['total_keywords']
                                    intensity = min(1.0, proportion + 0.2)
                                else:
                                    intensity = 0.8
                                color_intensity = int(intensity * 255)
                                st.markdown(
                                    f"""<div style="
                                        background-color: rgba({255-color_intensity//2}, 255, {255-color_intensity//2}, 0.3);
                                        padding: 8px;
                                        border-radius: 5px;
                                        border-left: 4px solid #44ff44;
                                        margin: 2px 0;
                                    ">
                                        <strong>Cap. {capitulo['numero']}</strong><br>
                                        {titulo_html}<br>
                                        <small>Mejor {'distancia' if similarity_method == 'euclidean' else 'similitud'}: {distancia:.3f}{info_keywords}{info_mejor_keyword}</small>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"""<div style="
                                        background-color: rgba(240, 240, 240, 0.3);
                                        padding: 6px;
                                        border-radius: 3px;
                                        margin: 2px 0;
                                        opacity: 0.7;
                                    ">
                                        Cap. {capitulo['numero']}: {titulo_html}<br>
                                        <small>Mejor {'distancia' if similarity_method == 'euclidean' else 'similitud'}: {distancia:.3f}{info_keywords}{info_mejor_keyword}</small>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
        # Mostrar top 10 más similares
        st.markdown("### 🏆 Top 10 Capítulos Más Similares (Mejor Keyword Match)")
        top_10 = df_distancias.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            label = "distancia" if similarity_method == "euclidean" else "similitud"
            distancia_text = f"Mejor {label}: {row['distancia']:.3f}"
            titulo_html = remove_accents_html(row['etiqueta']) if 'etiqueta' in row else ''
            mejor_keyword_html = remove_accents_html(row['mejor_keyword']) if 'mejor_keyword' in row else ''
            with st.expander(f"{i}. {titulo_html} ({distancia_text})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Curso:** {row['curso']}")
                    st.write(f"**{distancia_text}**")
                    if 'es_relevante' in row:
                        relevante_text = "✅ Relevante" if row['es_relevante'] else "❌ No relevante"
                        st.write(f"**Estado:** {relevante_text}")
                    if mostrar_cant_keywords and 'keywords_relevantes' in row and 'total_keywords' in row:
                        st.write(f"**Keywords relevantes:** {row['keywords_relevantes']}/{row['total_keywords']}")
                    if mostrar_mejor_keyword and 'mejor_keyword' in row:
                        st.write(f"**Keyword más similar:** {mejor_keyword_html}")
                with col2:
                    st.write("**Keywords del capítulo:**")
                    st.write(row['keywords'])
        # Opción para descargar resultados
        if st.button("💾 Descargar Resultados"):
            csv = df_distancias.to_csv(index=False)
            safe_prompt = safe_filename(prompt[:30].replace(' ', '_'))
            file_name_busqueda = f"busqueda_semantica_{similarity_method}_{safe_filename(safe_prompt)}.csv"
            debug_string("file_name_busqueda", file_name_busqueda)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=file_name_busqueda,
                mime="text/csv"
            )

def safe_filename(input_str):
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    no_accents = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', no_accents)
    cleaned = re.sub(r'_+', '_', cleaned)  # Reemplaza múltiples guiones bajos por uno solo
    cleaned = cleaned.strip('_-')  # Quita guiones bajos o guiones al inicio/fin
    if not cleaned:
        cleaned = "busqueda"
    return cleaned

def remove_accents_html(input_str):
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def main():
    """Función principal de la aplicación"""
    # Título principal
    st.title("🎓 CMM-EDU Visualizador")
    st.markdown("Herramientas de análisis y visualización para contenido educativo")
    
    # Sistema de pestañas
    tab1, tab2, tab5 = st.tabs(["📊 Embeddings", "🔥 Similitud", "🔍 Búsqueda Semántica"])
    
    with tab1:
        embeddings_tab()
    
    with tab2:
        similarity_heatmaps_tab()
    
    with tab5:
        semantic_search_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Desarrollado para CMM-EDU | Visualización de Embeddings Educativos
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 