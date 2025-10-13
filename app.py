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
import io
import ast

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
    st.plotly_chart(fig, use_container_width=True, key="embeddings_tab_main_plot")
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
            st.plotly_chart(fig_matrix, use_container_width=True, key="similarity_tab_matrix")
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
        st.plotly_chart(fig_avg, use_container_width=True, key="similarity_tab_avg")
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
            st.plotly_chart(fig_keywords, use_container_width=True, key="fig_keywords_1")
        
        with col2:
            fig_length = px.box(
                combined_metrics,
                x='curso',
                y='avg_keyword_length',
                title="Distribución de Longitud Promedio de Keywords por Curso",
                labels={'avg_keyword_length': 'Longitud Promedio Keywords'}
            )
            fig_length.update_layout(xaxis=dict(tickangle=45))
            st.plotly_chart(fig_length, use_container_width=True, key="fig_length_1")
    
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
            st.plotly_chart(fig_magnitude, use_container_width=True, key="fig_magnitude_1")
        
        with col2:
            fig_entropy = px.histogram(
                combined_metrics,
                x='embedding_entropy',
                color='curso',
                title="Distribución de Entropía de Embeddings",
                labels={'embedding_entropy': 'Entropía del Embedding'},
                marginal="box"
            )
            st.plotly_chart(fig_entropy, use_container_width=True, key="fig_entropy_1")
    
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
        
        st.plotly_chart(fig_corr, use_container_width=True, key="fig_corr_1")
        
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

# Extra tab with threshold slider
def similarity_heatmaps_tab_with_threshold():
    """Pestaña para heatmaps de similitud con umbral"""
    st.markdown('<h1 class="main-header">🧱 Umbrales de Similitud</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el archivo CSV esté en la carpeta 'data/'.")
        return
    
    # --- CONFIGURACIÓN DE SIMILITUD CON UMBRAL ---
    st.subheader("Configuración de Similitud")
    
    # Primero crear 2 columnas para los selectores principales
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
            key="similarity_heatmaps_method_threshold_unique"
        )
    
    cursos_disponibles = ['Todos'] + ordenar_cursos_personalizado(df['curso'].unique().tolist())
    with col_sel2:
        curso_seleccionado = st.selectbox(
            "Filtrar por Curso:",
            cursos_disponibles,
            help="Filtrar matriz por curso específico",
            key="sim_heatmaps_curso_threshold_unique"
        )
    
    # Luego el umbral en una sección separada
    st.subheader("Umbral de Similitud")
    
    # Control deslizante para el umbral de similitud
    if similarity_method == 'euclidean':
        # Para distancia euclidiana: menor valor = más similar
        threshold = st.slider(
            "Umbral de Distancia Máxima:",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.01,
            help="Solo se mostrarán conexiones con distancia menor o igual a este valor (valores más bajos = más similares)",
            key="threshold_slider_unique"
        )
    else:
        # Para coseno y producto punto: mayor valor = más similar
        threshold = st.slider(
            "Umbral de Similitud Mínima:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Solo se mostrarán conexiones con similitud mayor o igual a este valor",
            key="threshold_slider_unique"
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
    col1, col2, col3, col4 = st.columns(4)
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
    with col4:
        if similarity_method == 'euclidean':
            st.metric("Umbral Distancia", f"{threshold:.2f}")
        else:
            st.metric("Umbral Similitud", f"{threshold:.2f}")
    
    # Información adicional sobre los datos
    if len(df_filtrado) != len(df_unique):
        st.info(f"ℹ️ Se detectaron {len(df_filtrado)} filas totales, usando {len(df_unique)} capítulos únicos para el análisis.")
    
    # Calcular matriz de similitud
    with st.spinner(f"Calculando matriz de similitud usando {similarity_method}..."):
        similarity_matrix = calculate_similarity_matrix(embeddings_matrix, similarity_method)
    
    # --- NUEVA FUNCIONALIDAD: MATRIZ CON UMBRAL ---
    st.subheader("🎯 Matriz de Adyacencia con Umbral")
    
    # Crear matriz binaria basada en el umbral
    if similarity_method == 'euclidean':
        # Para distancia: valores por debajo del umbral = similares (1), otros = 0
        adjacency_matrix = (similarity_matrix <= threshold).astype(int)
        # Excluir diagonal (autosimilitud)
        np.fill_diagonal(adjacency_matrix, 0)
    else:
        # Para similitud: valores por encima del umbral = similares (1), otros = 0
        adjacency_matrix = (similarity_matrix >= threshold).astype(int)
        # Excluir diagonal (autosimilitud)
        np.fill_diagonal(adjacency_matrix, 0)
    
    # Mostrar estadísticas de la matriz de adyacencia
    total_possible_connections = len(similarity_matrix) * (len(similarity_matrix) - 1) / 2
    actual_connections = np.sum(adjacency_matrix) / 2  # Dividir por 2 porque es simétrica
    
    col_adj1, col_adj2, col_adj3 = st.columns(3)
    with col_adj1:
        st.metric("Conexiones Identificadas", int(actual_connections))
    with col_adj2:
        percentage = (actual_connections / total_possible_connections * 100) if total_possible_connections > 0 else 0
        st.metric("Porcentaje de Conexiones", f"{percentage:.1f}%")
    with col_adj3:
        densidad = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        st.metric("Densidad de la Red", f"{densidad:.3f}")
    
    # Crear y mostrar heatmap de la matriz de adyacencia
    try:
        method_names = {
            'cosine': 'Similitud Coseno',
            'euclidean': 'Distancia Euclidiana',
            'dot_product': 'Producto Punto'
        }
        
        threshold_type = "mínima de similitud" if similarity_method != 'euclidean' else "máxima de distancia"
        fig_adjacency = create_similarity_heatmap(
            adjacency_matrix, 
            labels, 
            f"Matriz de Adyacencia ({method_names[similarity_method]}) - Umbral {threshold_type}: {threshold}",
            'binary'  # Usar escala binaria para la matriz de adyacencia
        )
        
        if fig_adjacency is not None:
            st.plotly_chart(fig_adjacency, use_container_width=True, key="fig_adjacency_1")
        else:
            st.error("Error: create_similarity_heatmap devolvió None para la matriz de adyacencia")
            
    except Exception as e:
        st.error(f"Error creando matriz de adyacencia: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # --- NUEVA FUNCIONALIDAD: DESCARGAR MATRIZ DE ADYACENCIA ---
    st.subheader("📥 Descargar Matriz de Adyacencia")
    
    # Crear DataFrame de la matriz de adyacencia
    adj_df = pd.DataFrame(adjacency_matrix, index=labels, columns=labels)
    
    # Crear lista de conexiones para descarga
    connections_list = []
    for i in range(len(adjacency_matrix)):
        for j in range(i+1, len(adjacency_matrix)):  # Solo la mitad superior para evitar duplicados
            if adjacency_matrix[i, j] == 1:
                connections_list.append({
                    'curso_capitulo_1': courses[i],
                    'capitulo_1': labels[i],
                    'curso_capitulo_2': courses[j],
                    'capitulo_2': labels[j],
                    'similitud_original': similarity_matrix[i, j],
                    'umbral_aplicado': threshold
                })
    
    connections_df = pd.DataFrame(connections_list)
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        # Descargar matriz de adyacencia completa
        if st.button("💾 Matriz de Adyacencia (CSV)", key="download_adj_btn_threshold_unique"):
            csv_adj = adj_df.to_csv()
            file_name_adj = f"matriz_adyacencia_{similarity_method}_umbral{threshold:.2f}_{safe_filename(curso_seleccionado)}.csv"
            st.download_button(
                label="Descargar Matriz CSV",
                data=csv_adj,
                file_name=file_name_adj,
                mime="text/csv",
                key="download_adj_matrix_threshold_unique"
            )
    
    with col_download2:
        # Descargar lista de conexiones
        if st.button("📊 Lista de Conexiones (CSV)", key="download_conn_btn_threshold_unique"):
            if not connections_list:
                st.warning("No hay conexiones que cumplan con el umbral seleccionado.")
            else:
                csv_connections = connections_df.to_csv(index=False)
                file_name_conn = f"conexiones_{similarity_method}_umbral{threshold:.2f}_{safe_filename(curso_seleccionado)}.csv"
                st.download_button(
                    label="Descargar Conexiones CSV",
                    data=csv_connections,
                    file_name=file_name_conn,
                    mime="text/csv",
                    key="download_connections_threshold_unique"
                )
    
    
    # Mostrar vista previa de las conexiones
    if connections_list:
        st.subheader("🔍 Vista Previa de Conexiones Identificadas")
        st.write(f"**Se encontraron {len(connections_list)} conexiones con el umbral aplicado:**")
        
        # Mostrar las primeras 10 conexiones
        display_connections = connections_df.head(10).copy()
        if similarity_method == 'euclidean':
            display_connections = display_connections.rename(columns={'similitud_original': 'distancia_original'})
        
        st.dataframe(display_connections, use_container_width=True)
        

        # --- NUEVO DATAFRAME: CONEXIONES AGRUPADAS POR CAPÍTULO ---
        st.subheader("📋 Conexiones por Capítulo (Ordenadas por Similitud)")

        # --- CARGAR CONTEOS DESDE CSV ---
        keywords_df = pd.read_csv("data/capitulos_keywords_count.csv")

        keywords_df['capitulo_id'] = keywords_df.apply(capitulo_id, axis=1)

        # --- CONEXIONES POR CAPÍTULO ---
        capitulo_conexiones = {}

        for conn in connections_list:
            cap1 = conn['capitulo_1']
            cap2 = conn['capitulo_2']
            curso1 = conn['curso_capitulo_1']
            curso2 = conn['curso_capitulo_2']
            similitud = conn['similitud_original']
            
            if cap1 not in capitulo_conexiones:
                capitulo_conexiones[cap1] = {
                    'curso': curso1,
                    'capitulos_relacionados': [],
                    'similitudes': []
                }
            capitulo_conexiones[cap1]['capitulos_relacionados'].append(cap2)
            capitulo_conexiones[cap1]['similitudes'].append(similitud)
            
            if cap2 not in capitulo_conexiones:
                capitulo_conexiones[cap2] = {
                    'curso': curso2,
                    'capitulos_relacionados': [],
                    'similitudes': []
                }
            capitulo_conexiones[cap2]['capitulos_relacionados'].append(cap1)
            capitulo_conexiones[cap2]['similitudes'].append(similitud)

        # Ordenar conexiones
        for cap, data in capitulo_conexiones.items():
            combined = list(zip(data['capitulos_relacionados'], data['similitudes']))
            if similarity_method == 'euclidean':
                combined_sorted = sorted(combined, key=lambda x: x[1])
            else:
                combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
            data['capitulos_relacionados'] = [item[0] for item in combined_sorted]
            data['similitudes'] = [item[1] for item in combined_sorted]

        # Crear DataFrame final
        conexiones_por_capitulo = []
        for cap, data in capitulo_conexiones.items():
            # Buscar keywords principal
            keywords_principal = keywords_df.loc[
                keywords_df['capitulo_id'] == cap, 'num_keywords'
            ].values
            keywords_principal = int(keywords_principal[0]) if len(keywords_principal) > 0 else 0

            # Buscar keywords relacionados
            keywords_relacionados = []
            for cap_relacionado in data['capitulos_relacionados']:
                kw = keywords_df.loc[
                    keywords_df['capitulo_id'] == cap_relacionado, 'num_keywords'
                ].values
                keywords_relacionados.append(str(int(kw[0])) if len(kw) > 0 else "0")
            
            conexiones_por_capitulo.append({
                'curso': data['curso'],
                'capitulo': cap,
                'keywords_principal': keywords_principal,
                'capitulos_relacionados': ', '.join(data['capitulos_relacionados']),
                'keywords_relacionados': ', '.join(keywords_relacionados),
                'similitudes': ', '.join([f"{sim:.4f}" for sim in data['similitudes']])
            })

        df_conexiones_por_capitulo = pd.DataFrame(conexiones_por_capitulo)

        # Mostrar
        st.write("**Conexiones agrupadas por capítulo (ordenadas por similitud):**")
        st.dataframe(df_conexiones_por_capitulo, use_container_width=True, height=400)

        st.download_button(
            label="📥 Descargar Conexiones por Capítulo (CSV)",
            data=df_conexiones_por_capitulo.to_csv(index=False),
            file_name=f"conexiones_por_capitulo_{similarity_method}_umbral{threshold:.2f}.csv",
            mime="text/csv",
            key="download_conexiones_por_capitulo"
        )

        
        # --- ANÁLISIS ADICIONAL MEJORADO ---
        st.subheader("📊 Análisis de las Conexiones")
        
        col_anal1, col_anal2 = st.columns(2)
        
        with col_anal1:
            # Distribución de conexiones por curso - MEJORADO
            if len(connections_list) > 0:
                course_connections = []
                for conn in connections_list:
                    course_connections.extend([conn['curso_capitulo_1'], conn['curso_capitulo_2']])
                
                course_counts = pd.Series(course_connections).value_counts().reset_index()
                course_counts.columns = ['Curso', 'Conexiones']
                
                fig_courses = px.bar(
                    course_counts,
                    x='Curso',
                    y='Conexiones',
                    title="📚 Distribución de Conexiones por Curso",
                    labels={'Conexiones': 'Número de Conexiones', 'Curso': 'Curso'},
                    color='Conexiones',
                    color_continuous_scale='viridis',
                    text='Conexiones'
                )
                
                fig_courses.update_layout(
                    xaxis_tickangle=45,
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(248,249,250,1)',
                    font=dict(size=12, family='Arial'),
                    title_font=dict(size=16, family='Arial', color='#2c3e50'),
                    showlegend=False,
                    height=400
                )
                
                fig_courses.update_traces(
                    textposition='outside',
                    marker_line_color='black',
                    marker_line_width=1,
                    hovertemplate='<b>%{x}</b><br>Conexiones: %{y}<extra></extra>'
                )
                
                fig_courses.update_xaxes(
                    showgrid=False,
                    linecolor='black',
                    linewidth=1
                )
                
                fig_courses.update_yaxes(
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    linewidth=1
                )
                
                st.plotly_chart(fig_courses, use_container_width=True, key="fig_courses_1")
        
        with col_anal2:
            # Histograma de valores de similitud/distancia - MEJORADO
            if len(connections_list) > 0:
                sim_values = [conn['similitud_original'] for conn in connections_list]
                
                fig_hist = px.histogram(
                    x=sim_values,
                    title=f"📊 Distribución de {'Distancias' if similarity_method == 'euclidean' else 'Similitudes'}",
                    labels={'x': f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}", 
                        'y': 'Frecuencia'},
                    nbins=20,
                    color_discrete_sequence=['#1f77b4'],
                    opacity=0.8,
                    marginal="box"  # Agrega un box plot en el margen
                )
                
                # Agregar línea vertical para el umbral
                fig_hist.add_vline(
                    x=threshold, 
                    line_dash="dash", 
                    line_color="red",
                    line_width=3,
                    annotation_text=f"Umbral: {threshold:.2f}",
                    annotation_position="top right",
                    annotation_font=dict(color="red", size=12)
                )
                
                # Agregar línea de media
                mean_val = np.mean(sim_values)
                fig_hist.add_vline(
                    x=mean_val, 
                    line_dash="dot", 
                    line_color="green",
                    line_width=2,
                    annotation_text=f"Media: {mean_val:.2f}",
                    annotation_position="top left",
                    annotation_font=dict(color="green", size=10)
                )
                
                fig_hist.update_layout(
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(248,249,250,1)',
                    font=dict(size=12, family='Arial'),
                    title_font=dict(size=16, family='Arial', color='#2c3e50'),
                    height=400,
                    showlegend=False
                )
                
                fig_hist.update_traces(
                    marker_line_color='black',
                    marker_line_width=1,
                    hovertemplate=f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}: %{{x}}<br>Frecuencia: %{{y}}<extra></extra>"
                )
                
                fig_hist.update_xaxes(
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    linewidth=1,
                    title_font=dict(size=12)
                )
                
                fig_hist.update_yaxes(
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    linewidth=1,
                    title_font=dict(size=12)
                )
                
                st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist_1")
        
        # --- GRÁFICO ADICIONAL: TIPOS DE CONEXIÓN ---
        st.subheader("🔗 Análisis de Tipos de Conexión")
        
        col_anal3, col_anal4 = st.columns(2)
        
        with col_anal3:
            # Gráfico de pie para tipos de conexión (intra-curso vs inter-curso)
            if len(connections_list) > 0:
                intra_curso = len([c for c in connections_list if c['curso_capitulo_1'] == c['curso_capitulo_2']])
                inter_curso = len(connections_list) - intra_curso
                
                fig_pie = px.pie(
                    values=[intra_curso, inter_curso],
                    names=['Intra-curso', 'Inter-curso'],
                    title="🎯 Tipos de Conexiones",
                    color=['Intra-curso', 'Inter-curso'],
                    color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'}
                )
                
                fig_pie.update_traces(
                    textinfo='percent+label',
                    pull=[0.05, 0],
                    marker=dict(line=dict(color='white', width=2)),
                    hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
                )
                
                fig_pie.update_layout(
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(248,249,250,1)',
                    font=dict(size=12, family='Arial'),
                    title_font=dict(size=16, family='Arial', color='#2c3e50'),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    height=350
                )
                
                st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie_1")
        
        with col_anal4:
            if len(connections_list) > 0:
                # Preparar datos
                scatter_data = []
                for conn in connections_list:
                    scatter_data.append({
                        'similitud': conn['similitud_original'],
                        'tipo': 'Intra-curso' if conn['curso_capitulo_1'] == conn['curso_capitulo_2'] else 'Inter-curso',
                        'curso1': conn['curso_capitulo_1'],
                        'curso2': conn['curso_capitulo_2']
                    })

                df_scatter = pd.DataFrame(scatter_data)

                # Gráfico violin con densidad
                fig_violin = px.violin(
                    df_scatter,
                    x='tipo',
                    y='similitud',
                    color='tipo',
                    box=True,              # Agrega boxplot dentro
                    points="all",          # Muestra puntos individuales
                    title="🎻 Distribución de Similitud por Tipo de Conexión",
                    labels={'similitud': f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}",
                            'tipo': 'Tipo de Conexión'},
                    color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'}
                )

                # Agregar línea de umbral
                fig_violin.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Umbral: {threshold:.2f}"
                )

                fig_violin.update_layout(
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(248,249,250,1)',
                    font=dict(size=12, family='Arial'),
                    title_font=dict(size=16, family='Arial', color='#2c3e50'),
                    showlegend=False,
                    height=400
                )

                st.plotly_chart(fig_violin, use_container_width=True, key="fig_violin_1")

        
        # --- ESTADÍSTICAS RÁPIDAS ---
        st.subheader("📈 Estadísticas Rápidas")
        
        if len(connections_list) > 0:
            sim_values = [conn['similitud_original'] for conn in connections_list]
            intra_curso = len([c for c in connections_list if c['curso_capitulo_1'] == c['curso_capitulo_2']])
            inter_curso = len(connections_list) - intra_curso
            
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            with col_stats1:
                st.metric(
                    label="📊 Conexiones Totales",
                    value=len(connections_list),
                    delta=f"{intra_curso} intra-curso"
                )
            
            with col_stats2:
                st.metric(
                    label="🔗 Conexiones Inter-curso",
                    value=inter_curso,
                    delta=f"{(inter_curso/len(connections_list)*100):.1f}%"
                )
            
            with col_stats3:
                st.metric(
                    label=f"📏 {'Distancia' if similarity_method == 'euclidean' else 'Similitud'} Promedio",
                    value=f"{np.mean(sim_values):.3f}",
                    delta=f"±{np.std(sim_values):.3f}"
                )
            
            with col_stats4:
                st.metric(
                    label="🎯 Densidad de Red",
                    value=f"{densidad:.3f}",
                    delta=f"{(densidad*100):.1f}%"
                )

    else:
        st.warning("No se encontraron conexiones que cumplan con el umbral seleccionado. Intenta ajustar el umbral.")
    
    # --- RESTO DEL CÓDIGO ORIGINAL CON KEYS ÚNICOS ---
    
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
            st.plotly_chart(fig_matrix, use_container_width=True, key="capitulo_id_similarity_tab_matrix")
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
        st.plotly_chart(fig_avg, use_container_width=True, key="capitulo_id_similarity_tab_avg")
        st.write(f"Debug: Se creó gráfico con {len(df_avg)} capítulos")
        
    except Exception as e:
        st.error(f"Error creando gráfico promedio: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # ... (Aquí continuaría el resto del código original, pero necesitarías agregar keys únicos a todos los widgets adicionales)
    
    # Para los botones finales de descarga, usar keys únicos:
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        if st.button("💾 Descargar Matriz de Similitud como CSV", key="download_matrix_btn_threshold_unique"):
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
                key="download_matrix_threshold_unique"
            )
    
    with col_download2:
        if st.button("📊 Descargar Tabla de Métricas como CSV", key="download_metrics_btn_threshold_unique"):
            # Asumiendo que filtered_table existe del código anterior
            csv_metrics = filtered_table.to_csv(index=False) if 'filtered_table' in locals() else pd.DataFrame().to_csv()
            file_name_metrics = f"metricas_completas_{similarity_method}_{safe_filename(curso_seleccionado)}.csv"
            debug_string("file_name_metrics", file_name_metrics)
            st.download_button(
                label="Descargar Métricas CSV",
                data=csv_metrics,
                file_name=file_name_metrics,
                mime="text/csv",
                key="download_metrics_threshold_unique"
            )

# Extra tab for percentiles and thresholds
def similarity_heatmaps_tab_with_percentiles():
    """Pestaña para heatmaps de similitud con percentiles y umbrales móviles"""
    st.markdown('<h1 class="main-header">🎯 Percentiles y Umbrales de similitud</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el archivo CSV esté en la carpeta 'data/'.")
        return
    
    # --- FUNCIONES AUXILIARES ---
    def crear_capitulo_id(row):
        """Crear identificador único de capítulo"""
        curso_abbr = str(row['curso']).replace('Primero', '1B').replace('Segundo', '2B')\
                                    .replace('Tercero', '3B').replace('Cuarto', '4B')\
                                    .replace('Quinto', '5B').replace('Sexto', '6B')
        return f"{curso_abbr}: Capítulo N°{row['numero']}: {row['titulo']}"
    
    def cargar_keywords():
        """Cargar y preparar datos de keywords"""
        try:
            keywords_df = pd.read_csv("data/capitulos_keywords_count.csv")
            keywords_dict = keywords_df.set_index('id')['num_keywords'].to_dict()
            keywords_df['capitulo_id'] = keywords_df.apply(crear_capitulo_id, axis=1)
            id_to_capitulo_dict = keywords_df.set_index('id')['capitulo_id'].to_dict()
            return keywords_dict, id_to_capitulo_dict
        except Exception as e:
            st.warning(f"No se pudo cargar el archivo de keywords: {e}")
            return {}, {}
    
    def crear_adjacency_matrix(similarity_matrix, similarity_method, use_percentil, 
                             use_umbral, threshold_percentil, umbral_absoluto, es_modo_and):
        """Crear matriz de adyacencia basada en los filtros aplicados"""
        # Crear máscaras individuales
        mascara_percentil, mascara_umbral = None, None
        
        if use_percentil:
            mascara_percentil = (similarity_matrix <= threshold_percentil) if similarity_method == 'euclidean' else (similarity_matrix >= threshold_percentil)
        
        if use_umbral:
            mascara_umbral = (similarity_matrix <= umbral_absoluto) if similarity_method == 'euclidean' else (similarity_matrix >= umbral_absoluto)
        
        # Combinar máscaras según el modo
        if use_percentil and use_umbral:
            adjacency_matrix = (mascara_percentil & mascara_umbral).astype(int) if es_modo_and else (mascara_percentil | mascara_umbral).astype(int)
        elif use_percentil:
            adjacency_matrix = mascara_percentil.astype(int)
        elif use_umbral:
            adjacency_matrix = mascara_umbral.astype(int)
        else:
            adjacency_matrix = np.ones_like(similarity_matrix)
        
        # Excluir diagonal
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix
    
    def crear_conexiones_por_capitulo(connections_list, df_unique, similarity_method, keywords_dict):
        """Crear DataFrame de conexiones agrupadas por capítulo"""
        capitulo_conexiones = {}
        
        for conn in connections_list:
            cap1, cap2 = conn['capitulo_1'], conn['capitulo_2']
            curso1, curso2 = conn['curso_capitulo_1'], conn['curso_capitulo_2']
            similitud = conn['similitud_original']
            
            for cap, curso in [(cap1, curso1), (cap2, curso2)]:
                if cap not in capitulo_conexiones:
                    capitulo_conexiones[cap] = {
                        'curso': curso,
                        'capitulos_relacionados': [],
                        'similitudes': []
                    }
                capitulo_conexiones[cap]['capitulos_relacionados'].append(cap2 if cap == cap1 else cap1)
                capitulo_conexiones[cap]['similitudes'].append(similitud)
        
        # Ordenar conexiones por similitud
        for cap, data in capitulo_conexiones.items():
            combined = list(zip(data['capitulos_relacionados'], data['similitudes']))
            reverse = similarity_method != 'euclidean'
            combined_sorted = sorted(combined, key=lambda x: x[1], reverse=reverse)
            data['capitulos_relacionados'] = [item[0] for item in combined_sorted]
            data['similitudes'] = [item[1] for item in combined_sorted]
        
        # Crear DataFrame final
        conexiones_por_capitulo = []
        for cap, data in capitulo_conexiones.items():
            # Buscar número de capítulo y keywords
            numero_capitulo, cap_id_principal = 0, None
            for idx, row in df_unique.iterrows():
                capitulo_id_actual = crear_capitulo_id(row)
                if capitulo_id_actual == cap:
                    numero_capitulo, cap_id_principal = row['numero'], row['id']
                    break
            
            keywords_principal = keywords_dict.get(cap_id_principal, 0) if cap_id_principal else 0
            
            # Obtener keywords de capítulos relacionados
            keywords_relacionados_list = []
            for cap_relacionado in data['capitulos_relacionados']:
                cap_id_relacionado = None
                for idx, row in df_unique.iterrows():
                    if crear_capitulo_id(row) == cap_relacionado:
                        cap_id_relacionado = row['id']
                        break
                kw_rel = keywords_dict.get(cap_id_relacionado, 0) if cap_id_relacionado else 0
                keywords_relacionados_list.append(str(kw_rel))
            
            conexiones_por_capitulo.append({
                'curso': data['curso'],
                'capitulo': cap,
                'numero_capitulo': numero_capitulo,
                'keywords_principal': keywords_principal,
                'numero_conexiones': len(data['capitulos_relacionados']),
                'capitulos_relacionados': ', '.join(data['capitulos_relacionados']),
                'keywords_relacionados': ', '.join(keywords_relacionados_list),
                'similitudes': ', '.join([f"{sim:.4f}" for sim in data['similitudes']])
            })
        
        df_result = pd.DataFrame(conexiones_por_capitulo)
        df_result = df_result.sort_values(['curso', 'numero_capitulo'], ascending=[True, True])
        return df_result.drop('numero_capitulo', axis=1)
    
    def filtrar_conexiones_mayor_grado(connections_list):
        """Filtrar conexiones que involucren cursos de diferente grado"""
        orden_cursos = ['Primero Básico', 'Segundo Básico', 'Tercero Básico', 
                       'Cuarto Básico', 'Quinto Básico', 'Sexto Básico']
        
        return [conn for conn in connections_list 
                if orden_cursos.index(conn['curso_capitulo_1']) != orden_cursos.index(conn['curso_capitulo_2'])]
    
    # --- CONFIGURACIÓN DE SIMILITUD ---
    st.subheader("Configuración de Similitud")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        similarity_method = st.selectbox(
            "Método de Similitud:",
            ['cosine', 'euclidean', 'dot_product'],
            format_func=lambda x: {'cosine': 'Similitud Coseno', 'euclidean': 'Distancia Euclidiana', 
                                 'dot_product': 'Producto Punto'}[x],
            help="Selecciona el método para calcular similitud entre embeddings",
            key="similarity_percentiles_method"
        )
    
    cursos_disponibles = ['Todos'] + ordenar_cursos_personalizado(df['curso'].unique().tolist())
    with col_sel2:
        curso_seleccionado = st.selectbox(
            "Filtrar por Curso:", cursos_disponibles,
            help="Filtrar matriz por curso específico", key="sim_percentiles_curso"
        )
    
    # --- CONFIGURACIÓN DE UMBRALES Y PERCENTILES ---
    st.subheader("⚙️ Configuración de Umbrales")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        use_percentil = st.checkbox("Usar filtro por percentil", value=True, key="use_percentil")
        percentil = st.slider(
            "Percentil para conexiones más similares:", 1, 100, 20, 1,
            help="Solo se considerarán las conexiones que estén en el top X% de similitud",
            key="percentil_slider_main"
        ) if use_percentil else None
    
    with col_config2:
        use_umbral = st.checkbox("Usar filtro por umbral absoluto", value=True, key="use_umbral")
        if use_umbral:
            if similarity_method == 'euclidean':
                umbral_absoluto = st.slider("Umbral de distancia máxima:", 0.0, 2.0, 1.0, 0.01,
                                          help="Solo conexiones con distancia menor o igual a este valor",
                                          key="umbral_absoluto_slider")
            else:
                umbral_absoluto = st.slider("Umbral de similitud mínima:", 0.0, 1.0, 0.5, 0.01,
                                          help="Solo conexiones con similitud mayor o igual a este valor",
                                          key="umbral_absoluto_slider")
        else:
            umbral_absoluto = None
    
    # --- MODO DE COMBINACIÓN ---
    if use_percentil and use_umbral:
        st.write("**🔀 Modo de Combinación**")
        modo_combinacion = st.radio(
            "Cómo combinar los filtros:", ["AND (más restrictivo)", "OR (más permisivo)"], index=0,
            key="modo_combinacion", help="AND: debe cumplir AMBOS filtros | OR: debe cumplir AL MENOS UN filtro"
        )
        es_modo_and = (modo_combinacion == "AND (más restrictivo)")
    else:
        es_modo_and = False
    
    # --- PROCESAMIENTO DE DATOS ---
    df_filtrado = df if curso_seleccionado == 'Todos' else df[df['curso'] == curso_seleccionado]
    if len(df_filtrado) == 0:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return
    
    df_unique = df_filtrado.drop_duplicates(subset=['id']).copy()
    df_unique['capitulo_id'] = df_unique.apply(crear_capitulo_id, axis=1)
    
    labels = df_unique['capitulo_id'].tolist()
    embeddings_matrix = np.vstack(df_unique['embeddings_array'].values)
    courses = df_unique['curso'].tolist()
    
    # --- INFORMACIÓN GENERAL ---
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1: st.metric("Capítulos Únicos", len(df_unique))
    with col_info2: st.metric("Dimensión Embeddings", embeddings_matrix.shape[1])
    with col_info3: st.metric("Método", {'cosine': 'Coseno', 'euclidean': 'Euclidiana', 'dot_product': 'Producto Punto'}[similarity_method])
    with col_info4:
        if use_percentil and use_umbral: st.metric("Filtros", "Percentil + Umbral")
        elif use_percentil: st.metric("Filtro", f"Percentil {percentil}%")
        elif use_umbral: st.metric("Filtro", "Umbral Absoluto")
        else: st.metric("Filtro", "Ninguno")
    
    # --- CÁLCULO DE MATRICES ---
    with st.spinner(f"Calculando matriz de similitud usando {similarity_method}..."):
        similarity_matrix = calculate_similarity_matrix(embeddings_matrix, similarity_method)
    
    # --- CÁLCULO DE UMBRALES ---
    st.subheader("📈 Sobre la Configuración de Umbrales")
    
    all_similarities = [similarity_matrix[i, j] for i in range(len(similarity_matrix)) 
                       for j in range(i+1, len(similarity_matrix))]
    
    if not all_similarities:
        st.error("No se pudieron calcular similitudes. Verifica los datos.")
        return
    
    threshold_percentil = None
    if use_percentil:
        threshold_percentil = (np.percentile(all_similarities, percentil) if similarity_method == 'euclidean' 
                             else np.percentile(all_similarities, 100 - percentil))
        st.success(f"**Umbral del percentil {percentil}%:** {threshold_percentil:.4f}")
        st.info("🔍 Para distancias: valores **menores o iguales** a este umbral representan las conexiones más similares" 
                if similarity_method == 'euclidean' else 
                "🔍 Para similitudes: valores **mayores o iguales** a este umbral representan las conexiones más similares")
    
    if use_umbral:
        st.success(f"**Umbral absoluto:** {umbral_absoluto:.4f}")
        st.info("🔍 Para distancias: valores **menores o iguales** a este umbral" 
                if similarity_method == 'euclidean' else 
                "🔍 Para similitudes: valores **mayores o iguales** a este umbral")
    
    # --- MATRIZ DE ADYACENCIA ---
    st.subheader("🎯 Matriz de Adyacencia")
    
    adjacency_matrix = crear_adjacency_matrix(similarity_matrix, similarity_method, use_percentil, 
                                            use_umbral, threshold_percentil, umbral_absoluto, es_modo_and)
    
    # Información del modo aplicado
    if use_percentil and use_umbral:
        st.info("✅ Modo **AND**: Conexiones deben cumplir AMBOS filtros" if es_modo_and else 
                "✅ Modo **OR**: Conexiones deben cumplir AL MENOS UN filtro")
    elif use_percentil: st.info("✅ Usando solo filtro por **percentil**")
    elif use_umbral: st.info("✅ Usando solo filtro por **umbral absoluto**")
    else: st.info("ℹ️ Sin filtros aplicados - mostrando todas las conexiones posibles")
    
    # Estadísticas de la matriz
    total_possible_connections = len(similarity_matrix) * (len(similarity_matrix) - 1) / 2
    actual_connections = np.sum(adjacency_matrix) / 2
    densidad = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
    porcentaje = (actual_connections / total_possible_connections * 100) if total_possible_connections > 0 else 0
    conexiones_por_capitulo = actual_connections * 2 / len(similarity_matrix) if len(similarity_matrix) > 0 else 0
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1: st.metric("Conexiones Identificadas", int(actual_connections))
    with col_stats2: st.metric("Porcentaje de Conexiones", f"{porcentaje:.1f}%")
    with col_stats3: st.metric("Densidad de Red", f"{densidad:.4f}")
    with col_stats4: st.metric("Promedio Conexiones/Cap", f"{conexiones_por_capitulo:.1f}")
    
    # Mostrar heatmap
    method_names = {'cosine': 'Similitud Coseno', 'euclidean': 'Distancia Euclidiana', 'dot_product': 'Producto Punto'}
    
    if use_percentil and use_umbral:
        titulo = f"Matriz de Adyacencia - Percentil {percentil}% + Umbral {umbral_absoluto:.2f} ({method_names[similarity_method]})"
    elif use_percentil:
        titulo = f"Matriz de Adyacencia - Percentil {percentil}% ({method_names[similarity_method]})"
    elif use_umbral:
        titulo = f"Matriz de Adyacencia - Umbral {umbral_absoluto:.2f} ({method_names[similarity_method]})"
    else:
        titulo = f"Matriz de Adyacencia - Sin Filtros ({method_names[similarity_method]})"
    
    try:
        fig_adjacency = create_similarity_heatmap(adjacency_matrix, labels, titulo, 'binary')
        if fig_adjacency: st.plotly_chart(fig_adjacency, use_container_width=True, key="fig_adjacency_2")
    except Exception as e:
        st.error(f"Error creando matriz de adyacencia: {str(e)}")
    
    # --- LISTA DE CONEXIONES ---
    connections_list = []
    for i in range(len(adjacency_matrix)):
        for j in range(i+1, len(adjacency_matrix)):
            if adjacency_matrix[i, j] == 1:
                connections_list.append({
                    'curso_capitulo_1': courses[i], 'capitulo_1': labels[i],
                    'curso_capitulo_2': courses[j], 'capitulo_2': labels[j],
                    'similitud_original': similarity_matrix[i, j],
                    'percentil_aplicado': percentil if use_percentil else None,
                    'umbral_absoluto': umbral_absoluto if use_umbral else None,
                    'umbral_percentil': threshold_percentil if use_percentil else None
                })
    
    connections_df = pd.DataFrame(connections_list)
    
    # --- DESCARGAS PRINCIPALES ---
    st.subheader("📥 Descargar Matriz de Adyacencia")
    
    col_download1, col_download2 = st.columns(2)
    adj_df = pd.DataFrame(adjacency_matrix, index=labels, columns=labels)
    
    with col_download1:
        if st.button("💾 Matriz de Adyacencia (CSV)", key="download_adj_btn_percentiles"):
            csv_adj = adj_df.to_csv()
            file_name = f"matriz_adyacencia_percentil_{percentil if use_percentil else 'no'}_umbral_{umbral_absoluto if use_umbral else 'no'}_{similarity_method}.csv"
            st.download_button("Descargar Matriz CSV", csv_adj, file_name, "text/csv", key="download_adj_matrix_percentiles")
    
    with col_download2:
        if st.button("📊 Lista de Conexiones (CSV)", key="download_conn_btn_percentiles"):
            if connections_list:
                csv_connections = connections_df.to_csv(index=False)
                file_name = f"conexiones_percentil_{percentil if use_percentil else 'no'}_umbral_{umbral_absoluto if use_umbral else 'no'}_{similarity_method}.csv"
                st.download_button("Descargar Conexiones CSV", csv_connections, file_name, "text/csv", key="download_connections_percentiles")
            else:
                st.warning("No hay conexiones que cumplan con los filtros seleccionados.")
    
    # --- ANÁLISIS DETALLADO (solo si hay conexiones) ---
    if connections_list:
        st.subheader("🔍 Vista Previa de Conexiones Identificadas")
        st.write(f"**Se encontraron {len(connections_list)} conexiones con los filtros aplicados:**")
        
        display_connections = connections_df.head(10).copy()
        if similarity_method == 'euclidean':
            display_connections = display_connections.rename(columns={'similitud_original': 'distancia_original'})
        st.dataframe(display_connections, use_container_width=True)
        
        # --- CONEXIONES AGRUPADAS POR CAPÍTULO ---
        st.subheader("📋 Conexiones por Capítulo (Ordenadas por Curso y Capítulo)")
        
        keywords_dict, _ = cargar_keywords()
        df_conexiones_por_capitulo = crear_conexiones_por_capitulo(connections_list, df_unique, similarity_method, keywords_dict)
        
        st.write(f"**Conexiones agrupadas por capítulo ({len(df_conexiones_por_capitulo)} capítulos con conexiones):**")
        st.dataframe(df_conexiones_por_capitulo, use_container_width=True, height=400)
        
        # Estadísticas rápidas de conexiones
        if not df_conexiones_por_capitulo.empty:
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                avg_connections = df_conexiones_por_capitulo['numero_conexiones'].mean()
                st.metric("Promedio conexiones/capítulo", f"{avg_connections:.1f}")
            with col_stats2:
                max_connections = df_conexiones_por_capitulo['numero_conexiones'].max()
                st.metric("Máximo conexiones", max_connections)
            with col_stats3:
                min_connections = df_conexiones_por_capitulo['numero_conexiones'].min()
                st.metric("Mínimo conexiones", min_connections)
        
        # Descargar conexiones por capítulo
        if not df_conexiones_por_capitulo.empty:
            filename_parts = ["conexiones_por_capitulo"]
            if use_percentil: filename_parts.append(f"p{percentil}")
            if use_umbral: filename_parts.append(f"u{umbral_absoluto:.2f}")
            filename_parts.append(similarity_method)
            
            st.download_button(
                "📥 Descargar Conexiones por Capítulo (CSV)",
                df_conexiones_por_capitulo.to_csv(index=False),
                "_".join(filename_parts) + ".csv", "text/csv",
                key="download_conexiones_por_capitulo_percentiles_unique"
            )
        
        # --- ANÁLISIS GRÁFICO ---
        st.subheader("📊 Análisis de las Conexiones")
        
        col_anal1, col_anal2 = st.columns(2)
        
        with col_anal1:
            # Gráfico de distribución por curso con filtro
            st.write("**🎯 Filtro de Conexiones**")
            filtrar_mayor_grado = st.checkbox(
                "Mostrar solo conexiones con cursos de mayor grado", False,
                help="Si está activado, solo se considerarán conexiones donde al menos un curso sea de grado superior"
            )
            
            connections_list_filtrada = filtrar_conexiones_mayor_grado(connections_list) if filtrar_mayor_grado else connections_list
            titulo_grafico = "📚 Conexiones con Cursos de Mayor Grado" if filtrar_mayor_grado else "📚 Distribución de Conexiones por Curso"
            
            if connections_list_filtrada:
                course_connections = []
                for conn in connections_list_filtrada:
                    course_connections.extend([conn['curso_capitulo_1'], conn['curso_capitulo_2']])
                
                course_counts = pd.Series(course_connections).value_counts().reset_index()
                course_counts.columns = ['Curso', 'Conexiones']
                
                fig_courses = px.bar(course_counts, x='Curso', y='Conexiones', title=titulo_grafico,
                                   color='Conexiones', color_continuous_scale='viridis', text='Conexiones')
                fig_courses.update_layout(xaxis_tickangle=45, plot_bgcolor='rgba(248,249,250,1)',
                                        paper_bgcolor='rgba(248,249,250,1)', showlegend=False, height=400)
                fig_courses.update_traces(textposition='outside', marker_line_color='black', marker_line_width=1)
                st.plotly_chart(fig_courses, use_container_width=True, key="fig_courses_2")
                
                if filtrar_mayor_grado:
                    st.info(f"**Conexiones con mayor grado:** {len(connections_list_filtrada)} de {len(connections_list)} totales ({(len(connections_list_filtrada)/len(connections_list)*100):.1f}%)")
            else:
                st.warning("No hay conexiones que cumplan con el filtro seleccionado.")
        
        with col_anal2:
            # Histograma de similitudes
            if connections_list:
                sim_values = [conn['similitud_original'] for conn in connections_list]
                titulo_hist = f"📊 Distribución de {'Distancias' if similarity_method == 'euclidean' else 'Similitudes'}"
                
                fig_hist = px.histogram(x=sim_values, title=titulo_hist, nbins=20,
                                      labels={'x': f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}", 'y': 'Frecuencia'},
                                      color_discrete_sequence=['#1f77b4'], opacity=0.8, marginal="box")
                
                if use_percentil and threshold_percentil:
                    fig_hist.add_vline(x=threshold_percentil, line_dash="dash", line_color="red",
                                     annotation_text=f"Percentil {percentil}%")
                if use_umbral:
                    fig_hist.add_vline(x=umbral_absoluto, line_dash="dot", line_color="blue",
                                     annotation_text=f"Umbral: {umbral_absoluto:.2f}")
                
                fig_hist.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)',
                                     height=400, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist_2")
        
        # --- ANÁLISIS DE TIPOS DE CONEXIÓN ---
        st.subheader("🔗 Análisis de Tipos de Conexión")
        
        col_anal3, col_anal4 = st.columns(2)
        
        with col_anal3:
            # Gráfico de pie
            if connections_list:
                intra_curso = len([c for c in connections_list if c['curso_capitulo_1'] == c['curso_capitulo_2']])
                inter_curso = len(connections_list) - intra_curso
                
                fig_pie = px.pie(values=[intra_curso, inter_curso], names=['Intra-curso', 'Inter-curso'],
                               title="🎯 Tipos de Conexiones", color=['Intra-curso', 'Inter-curso'],
                               color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'})
                fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0],
                                    marker=dict(line=dict(color='white', width=2)))
                fig_pie.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)',
                                    height=350, showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie_2")
        
        with col_anal4:
            # Violin plot
            if connections_list:
                scatter_data = [{
                    'similitud': conn['similitud_original'],
                    'tipo': 'Intra-curso' if conn['curso_capitulo_1'] == conn['curso_capitulo_2'] else 'Inter-curso'
                } for conn in connections_list]
                
                df_scatter = pd.DataFrame(scatter_data)
                fig_violin = px.violin(df_scatter, x='tipo', y='similitud', color='tipo', box=True, points="all",
                                     title="🎻 Distribución por Tipo de Conexión",
                                     labels={'similitud': f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}"},
                                     color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'})
                
                if use_percentil and threshold_percentil:
                    fig_violin.add_hline(y=threshold_percentil, line_dash="dash", line_color="red", line_width=2,
                                       annotation_text=f"Percentil {percentil}%: {threshold_percentil:.4f}")
                if use_umbral:
                    fig_violin.add_hline(y=umbral_absoluto, line_dash="dot", line_color="blue", line_width=2,
                                       annotation_text=f"Umbral: {umbral_absoluto:.4f}")
                
                fig_violin.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)',
                                       height=350, showlegend=False)
                st.plotly_chart(fig_violin, use_container_width=True, key="fig_violin_2")
        
        # --- ESTADÍSTICAS RÁPIDAS ---
        st.subheader("📈 Estadísticas Rápidas")
        
        sim_values = [conn['similitud_original'] for conn in connections_list]
        intra_curso = len([c for c in connections_list if c['curso_capitulo_1'] == c['curso_capitulo_2']])
        inter_curso = len(connections_list) - intra_curso
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1: st.metric("📊 Conexiones Totales", len(connections_list), delta=f"{intra_curso} intra-curso")
        with col_stats2: st.metric("🔗 Conexiones Inter-curso", inter_curso, delta=f"{(inter_curso/len(connections_list)*100):.1f}%")
        with col_stats3: st.metric(f"📏 {'Distancia' if similarity_method == 'euclidean' else 'Similitud'} Promedio", 
                                 f"{np.mean(sim_values):.3f}", delta=f"±{np.std(sim_values):.3f}")
        with col_stats4: st.metric("🎯 Densidad de Red", f"{densidad:.3f}", delta=f"{(densidad*100):.1f}%")
    
    else:
        st.warning("No se encontraron conexiones que cumplan con el umbral seleccionado. Intenta ajustar el umbral.")
    
    # --- MATRIZ COMPLETA Y ESTADÍSTICAS FINALES ---
    st.subheader("🔥 Matriz de Similitud Completa (Referencia)")
    
    try:
        fig_matrix = create_similarity_heatmap(similarity_matrix, labels, 
                                             f"Matriz de Similitud Completa ({method_names[similarity_method]})", 
                                             similarity_method)
        if fig_matrix: st.plotly_chart(fig_matrix, use_container_width=True, key="fig_matrix_1")
    except Exception as e:
        st.error(f"Error creando matriz de similitud completa: {str(e)}")
    
    # Estadísticas finales
    st.subheader("📊 Estadísticas de la Matriz de Similitud")
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    col1, col2, col3, col4 = st.columns(4)
    if similarity_method == 'euclidean':
        with col1: st.metric("Distancia Promedio", f"{np.mean(upper_triangle):.3f}")
        with col2: st.metric("Distancia Máxima", f"{np.max(upper_triangle):.3f}")
        with col3: st.metric("Distancia Mínima", f"{np.min(upper_triangle):.3f}")
        with col4: st.metric("Desviación Estándar", f"{np.std(upper_triangle):.3f}")
    else:
        with col1: st.metric("Similitud Promedio", f"{np.mean(upper_triangle):.3f}")
        with col2: st.metric("Similitud Máxima", f"{np.max(upper_triangle):.3f}")
        with col3: st.metric("Similitud Mínima", f"{np.min(upper_triangle):.3f}")
        with col4: st.metric("Desviación Estándar", f"{np.std(upper_triangle):.3f}")
    
    # --- DESCARGAS FINALES ---
    st.subheader("📥 Descargas Finales")
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if st.button("💾 Descargar Matriz de Similitud como CSV", key="download_matrix_btn_percentiles"):
            matrix_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
            st.download_button("Descargar Matriz CSV", matrix_df.to_csv(), 
                             f"matriz_similitud_{similarity_method}_percentiles.csv", "text/csv",
                             key="download_matrix_percentiles")
    
    with col_dl2:
        if st.button("📊 Descargar Estadísticas Completas", key="download_stats_btn_percentiles"):
            stats_data = {
                'Método': [similarity_method], 'Capítulos Analizados': [len(df_unique)],
                'Conexiones Totales': [len(connections_list) if connections_list else 0],
                'Densidad Red': [densidad], 'Percentil Aplicado': [percentil if use_percentil else 'No'],
                'Umbral Absoluto': [umbral_absoluto if use_umbral else 'No'],
                'Similitud Promedio': [np.mean(upper_triangle)], 'Similitud Máxima': [np.max(upper_triangle)],
                'Similitud Mínima': [np.min(upper_triangle)], 'Desviación Estándar': [np.std(upper_triangle)]
            }
            stats_df = pd.DataFrame(stats_data)
            st.download_button("Descargar Estadísticas CSV", stats_df.to_csv(index=False),
                             f"estadisticas_completas_percentiles_{similarity_method}.csv", "text/csv",
                             key="download_stats_percentiles")

# Extra tab for percentiles and thresholds - VERSION matrix keywords with percentiles
def similarity_heatmaps_tab_with_percentiles_keywords():
    """Pestaña para heatmaps de similitud con percentiles en las keywords y umbrales"""
    st.markdown('<h1 class="main-header">🎯 Método de Percentiles por Keywords</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el archivo CSV esté en la carpeta 'data/'.")
        return
    
    # --- FUNCIONES AUXILIARES ---
    def crear_capitulo_id(row):
        """Crear identificador único de capítulo"""
        curso_abbr = str(row['curso']).replace('Primero', '1B').replace('Segundo', '2B')\
                                    .replace('Tercero', '3B').replace('Cuarto', '4B')\
                                    .replace('Quinto', '5B').replace('Sexto', '6B')
        return f"{curso_abbr}: Capítulo N°{row['numero']}: {row['titulo']}"
    
    def cargar_keywords():
        """Cargar y preparar datos de keywords"""
        try:
            keywords_df = pd.read_csv("data/capitulos_keywords_count.csv")
            keywords_dict = keywords_df.set_index('id')['num_keywords'].to_dict()
            keywords_df['capitulo_id'] = keywords_df.apply(crear_capitulo_id, axis=1)
            id_to_capitulo_dict = keywords_df.set_index('id')['capitulo_id'].to_dict()
            return keywords_dict, id_to_capitulo_dict
        except Exception as e:
            st.warning(f"No se pudo cargar el archivo de keywords: {e}")
            return {}, {}
    
    def parse_embeddings(df):
        """Convertir strings de embeddings a listas de floats"""
        df = df.copy()
        df['keywords_embedding'] = df['keywords_embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))
        return df
    
    def calcular_matriz_percentil(df_unique, method="cosine", percentil=80):
        """Construir matriz de similitud filtrada por percentiles entre keywords de capítulos"""
        n = len(df_unique)
        mat = np.zeros((n, n))
        
        # Precompute todos los embeddings
        embeddings = []
        for i in range(n):
            emb = df_unique.iloc[i]['keywords_embedding']
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            embeddings.append(emb)
        
        for i in range(n):
            emb_i = embeddings[i]
            J_i = emb_i.shape[0]  # Número de keywords del capítulo i
            
            for j in range(i, n):
                emb_j = embeddings[j]
                J_j = emb_j.shape[0]  # Número de keywords del capítulo j
                
                # Calcular matriz de similitud entre TODAS las keywords
                if method == "cosine":
                    # Normalizar embeddings para cosine similarity
                    norms_i = np.linalg.norm(emb_i, axis=1, keepdims=True)
                    norms_j = np.linalg.norm(emb_j, axis=1, keepdims=True)
                    emb_i_norm = emb_i / np.where(norms_i == 0, 1, norms_i)
                    emb_j_norm = emb_j / np.where(norms_j == 0, 1, norms_j)
                    S_ij = emb_i_norm @ emb_j_norm.T
                elif method == "euclidean":
                    # Calcular matriz de distancias euclidianas
                    S_ij = np.sqrt(
                        np.sum(emb_i**2, axis=1)[:, None] + 
                        np.sum(emb_j**2, axis=1)[None, :] - 
                        2 * emb_i @ emb_j.T
                    )
                elif method == "dot_product":
                    S_ij = emb_i @ emb_j.T
                else:
                    raise ValueError("Método desconocido")
                
                # APLICAR METODOLOGÍA: Filtrar por percentil
                if i == j:
                    # Para el mismo capítulo, usar similitud máxima
                    mat[i, j] = 1.0 if method != "euclidean" else 0.0
                    mat[j, i] = mat[i, j]
                else:
                    # Para capítulos diferentes, aplicar filtrado por percentil
                    
                    # 1. Extraer valores para calcular el percentil
                    # Según especificación: considerar toda la matriz S_ij (ya que i≠j)
                    sim_vals = S_ij.flatten()
                    
                    if len(sim_vals) > 0:
                        # 2. Calcular umbral τ_p(i,j) = percentil(S_ij, p)
                        if method == "euclidean":
                            # Para distancias euclidianas: percentil inferior
                            # (conservar distancias pequeñas = más similares)
                            tau_p = np.percentile(sim_vals, 100 - percentil)
                            # Máscara binaria: B_ij = [S_ij <= τ_p]
                            mask_filtro = S_ij <= tau_p
                        else:
                            # Para similitudes (cosine, dot_product): percentil superior
                            # (conservar similitudes altas)
                            tau_p = np.percentile(sim_vals, percentil)
                            # Máscara binaria: B_ij = [S_ij >= τ_p]
                            mask_filtro = S_ij >= tau_p
                        
                        # 3. Aplicar filtrado: S̃_ij = S_ij ⊙ B_ij (producto de Hadamard)
                        S_ij_filtrado = S_ij * mask_filtro
                        
                        # 4. Calcular promedio: S̄_ij = Σ(S̃_ij) / ||B_ij||_0
                        # Solo sobre valores que cumplen la condición (B_ij = 1)
                        valores_no_cero = S_ij_filtrado[mask_filtro]
                        if len(valores_no_cero) > 0:
                            promedio = np.mean(valores_no_cero)
                        else:
                            promedio = 0.0
                    else:
                        promedio = 0.0
                    
                    mat[i, j] = promedio
                    mat[j, i] = promedio
        
        return mat
    
    def crear_conexiones_por_capitulo(connections_list, df_unique, similarity_method, keywords_dict):
        """Crear DataFrame de conexiones agrupadas por capítulo"""
        capitulo_conexiones = {}
        
        for conn in connections_list:
            cap1, cap2 = conn['capitulo_1'], conn['capitulo_2']
            curso1, curso2 = conn['curso_capitulo_1'], conn['curso_capitulo_2']
            similitud = conn['similitud_original']
            
            for cap, curso in [(cap1, curso1), (cap2, curso2)]:
                if cap not in capitulo_conexiones:
                    capitulo_conexiones[cap] = {
                        'curso': curso,
                        'capitulos_relacionados': [],
                        'similitudes': []
                    }
                capitulo_conexiones[cap]['capitulos_relacionados'].append(cap2 if cap == cap1 else cap1)
                capitulo_conexiones[cap]['similitudes'].append(similitud)
        
        # Ordenar conexiones por similitud
        for cap, data in capitulo_conexiones.items():
            combined = list(zip(data['capitulos_relacionados'], data['similitudes']))
            reverse = similarity_method != 'euclidean'
            combined_sorted = sorted(combined, key=lambda x: x[1], reverse=reverse)
            data['capitulos_relacionados'] = [item[0] for item in combined_sorted]
            data['similitudes'] = [item[1] for item in combined_sorted]
        
        # Crear DataFrame final
        conexiones_por_capitulo = []
        for cap, data in capitulo_conexiones.items():
            # Buscar número de capítulo y keywords
            numero_capitulo, cap_id_principal = 0, None
            for idx, row in df_unique.iterrows():
                capitulo_id_actual = crear_capitulo_id(row)
                if capitulo_id_actual == cap:
                    numero_capitulo, cap_id_principal = row['numero'], row['id']
                    break
            
            keywords_principal = keywords_dict.get(cap_id_principal, 0) if cap_id_principal else 0
            
            # Obtener keywords de capítulos relacionados
            keywords_relacionados_list = []
            for cap_relacionado in data['capitulos_relacionados']:
                cap_id_relacionado = None
                for idx, row in df_unique.iterrows():
                    if crear_capitulo_id(row) == cap_relacionado:
                        cap_id_relacionado = row['id']
                        break
                kw_rel = keywords_dict.get(cap_id_relacionado, 0) if cap_id_relacionado else 0
                keywords_relacionados_list.append(str(kw_rel))
            
            conexiones_por_capitulo.append({
                'curso': data['curso'],
                'capitulo': cap,
                'numero_capitulo': numero_capitulo,
                'keywords_principal': keywords_principal,
                'numero_conexiones': len(data['capitulos_relacionados']),
                'capitulos_relacionados': ', '.join(data['capitulos_relacionados']),
                'keywords_relacionados': ', '.join(keywords_relacionados_list),
                'similitudes': ', '.join([f"{sim:.4f}" for sim in data['similitudes']])
            })
        
        df_result = pd.DataFrame(conexiones_por_capitulo)
        df_result = df_result.sort_values(['curso', 'numero_capitulo'], ascending=[True, True])
        return df_result.drop('numero_capitulo', axis=1)
    
    def filtrar_conexiones_mayor_grado(connections_list):
        """Filtrar conexiones que involucren cursos de diferente grado"""
        orden_cursos = ['Primero Básico', 'Segundo Básico', 'Tercero Básico', 
                       'Cuarto Básico', 'Quinto Básico', 'Sexto Básico']
        
        return [conn for conn in connections_list 
                if orden_cursos.index(conn['curso_capitulo_1']) != orden_cursos.index(conn['curso_capitulo_2'])]
    
    # --- CONFIGURACIÓN DE SIMILITUD ---
    st.subheader("Configuración de Similitud")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        similarity_method = st.selectbox(
            "Método de Similitud:",
            ['cosine', 'euclidean', 'dot_product'],
            format_func=lambda x: {'cosine': 'Similitud Coseno', 'euclidean': 'Distancia Euclidiana', 
                                 'dot_product': 'Producto Punto'}[x],
            help="Selecciona el método para calcular similitud entre embeddings de keywords",
            key="similarity_percentiles_keywords_method"
        )
    
    cursos_disponibles = ['Todos'] + ordenar_cursos_personalizado(df['curso'].unique().tolist())
    with col_sel2:
        curso_seleccionado = st.selectbox(
            "Filtrar por Curso:", cursos_disponibles,
            help="Filtrar matriz por curso específico", key="sim_percentiles_keywords_curso"
        )
    
    # --- CONFIGURACIÓN DE PARÁMETROS ---
    st.subheader("🎯 Parámetros de la Metodología")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        percentil = st.slider(
            "Percentil p:", 0, 100, 80, 1,
            help="Conserva solo el top (100-p)% de las comparaciones más similares entre keywords",
            key="percentil_slider_keywords"
        )
        st.info(f"**p = {percentil}%**")
    
    with col_config2:
        if similarity_method == 'euclidean':
            threshold = st.slider(
                "Umbral r:", 0.0, 2.0, 1.0, 0.01,
                help="Capítulos relacionados si distancia promedio ≤ r",
                key="threshold_slider_keywords"
            )
        else:
            threshold = st.slider(
                "Umbral r:", 0.0, 1.0, 0.5, 0.01,
                help="Capítulos relacionados si similitud promedio ≥ r",
                key="threshold_slider_keywords"
            )
        st.info(f"**r = {threshold:.2f}**")
    
    # --- PROCESAMIENTO DE DATOS ---
    df_filtrado = df if curso_seleccionado == 'Todos' else df[df['curso'] == curso_seleccionado]
    if len(df_filtrado) == 0:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return
    
    df_unique = df_filtrado.drop_duplicates(subset=['id']).copy()
    df_unique = parse_embeddings(df_unique)
    df_unique['capitulo_id'] = df_unique.apply(crear_capitulo_id, axis=1)
    
    labels = df_unique['capitulo_id'].tolist()
    courses = df_unique['curso'].tolist()
    
    # --- INFORMACIÓN GENERAL ---
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1: st.metric("Capítulos Únicos", len(df_unique))
    with col_info2: st.metric("Dimensión Keywords", df_unique.iloc[0]['keywords_embedding'].shape[0])
    with col_info3: st.metric("Método", {'cosine': 'Coseno', 'euclidean': 'Euclidiana', 'dot_product': 'Producto Punto'}[similarity_method])
    with col_info4: st.metric("Configuración", f"p={percentil}%, r={threshold:.2f}")
    
    # --- CÁLCULO DE MATRICES ---
    st.subheader(f"🧮 Construyendo Matriz con Percentil p={percentil}%")
    with st.spinner("Calculando matriz de similitud filtrada por percentil..."):
        similarity_matrix = calcular_matriz_percentil(df_unique, similarity_method, percentil)
    
    # --- MATRIZ DE ADYACENCIA ---
    st.subheader("🎯 Matriz de Adyacencia")
    
    if similarity_method == "euclidean":
        adjacency_matrix = (similarity_matrix <= threshold).astype(int)
    else:
        adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)
    
    total_possible_keyword_pairs = 0
    actual_keyword_pairs = 0
    actual_connections = np.sum(adjacency_matrix) / 2
    
    for i in range(len(df_unique)):
        emb_i = df_unique.iloc[i]['keywords_embedding']
        J_i = emb_i.shape[0] if emb_i.ndim > 1 else 1
        for j in range(i+1, len(df_unique)):
            emb_j = df_unique.iloc[j]['keywords_embedding']  
            J_j = emb_j.shape[0] if emb_j.ndim > 1 else 1
            total_possible_keyword_pairs += J_i * J_j
            if adjacency_matrix[i, j] == 1:
                actual_keyword_pairs += 1

    # Métricas actualizadas
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1: 
        st.metric("Conexiones entre Capítulos", int(actual_connections))
    with col_stats2: 
        st.metric("Pares de Keywords Relacionados", actual_keyword_pairs)
    
    col_stats3, col_stats4 = st.columns(2)
    with col_stats3: 
        densidad_keywords = actual_keyword_pairs / total_possible_keyword_pairs if total_possible_keyword_pairs > 0 else 0
        st.metric("Densidad Keywords", f"{densidad_keywords:.4f}")
    with col_stats4: 
        st.metric("Configuración", f"p={percentil}%, r={threshold:.2f}")
    
    # Mostrar heatmap
    method_names = {'cosine': 'Similitud Coseno', 'euclidean': 'Distancia Euclidiana', 'dot_product': 'Producto Punto'}
    titulo = f"Matriz de Adyacencia - Percentil {percentil}% + Umbral {threshold:.2f} ({method_names[similarity_method]})"
    
    try:
        fig_adjacency = create_similarity_heatmap(adjacency_matrix, labels, titulo, 'binary')
        if fig_adjacency: st.plotly_chart(fig_adjacency, use_container_width=True, key="fig_adjacency_3")
    except Exception as e:
        st.error(f"Error creando matriz de adyacencia: {str(e)}")
    
    # --- LISTA DE CONEXIONES ---
    connections_list = []
    for i in range(len(adjacency_matrix)):
        for j in range(i+1, len(adjacency_matrix)):
            if adjacency_matrix[i, j] == 1:
                connections_list.append({
                    'curso_capitulo_1': courses[i], 'capitulo_1': labels[i],
                    'curso_capitulo_2': courses[j], 'capitulo_2': labels[j],
                    'similitud_original': similarity_matrix[i, j],
                    'percentil_aplicado': percentil,
                    'umbral_absoluto': threshold
                })
    
    connections_df = pd.DataFrame(connections_list)
    
    # --- DESCARGAS PRINCIPALES ---
    st.subheader("📥 Descargar Matriz de Adyacencia")
    
    col_download1, col_download2 = st.columns(2)
    adj_df = pd.DataFrame(adjacency_matrix, index=labels, columns=labels)
    
    with col_download1:
        if st.button("💾 Matriz de Adyacencia (CSV)", key="download_adj_btn_keywords"):
            csv_adj = adj_df.to_csv()
            file_name = f"matriz_adyacencia_keywords_p{percentil}_r{threshold:.2f}_{similarity_method}.csv"
            st.download_button("Descargar Matriz CSV", csv_adj, file_name, "text/csv", key="download_adj_matrix_keywords")
    
    with col_download2:
        if st.button("📊 Lista de Conexiones (CSV)", key="download_conn_btn_keywords"):
            if connections_list:
                csv_connections = connections_df.to_csv(index=False)
                file_name = f"conexiones_keywords_p{percentil}_r{threshold:.2f}_{similarity_method}.csv"
                st.download_button("Descargar Conexiones CSV", csv_connections, file_name, "text/csv", key="download_connections_keywords")
            else:
                st.warning("No hay conexiones que cumplan con los filtros seleccionados.")
    
    # --- ANÁLISIS DETALLADO (solo si hay conexiones) ---
    if connections_list:
        st.subheader("🔍 Vista Previa de Conexiones Identificadas")
        st.write(f"**Se encontraron {len(connections_list)} conexiones con los filtros aplicados:**")
        
        display_connections = connections_df.head(10).copy()
        if similarity_method == 'euclidean':
            display_connections = display_connections.rename(columns={'similitud_original': 'distancia_original'})
        st.dataframe(display_connections, use_container_width=True)
        
        # --- CONEXIONES AGRUPADAS POR CAPÍTULO ---
        st.subheader("📋 Conexiones por Capítulo (Ordenadas por Curso y Capítulo)")
        
        keywords_dict, _ = cargar_keywords()
        df_conexiones_por_capitulo = crear_conexiones_por_capitulo(connections_list, df_unique, similarity_method, keywords_dict)
        
        st.write(f"**Conexiones agrupadas por capítulo ({len(df_conexiones_por_capitulo)} capítulos con conexiones):**")
        st.dataframe(df_conexiones_por_capitulo, use_container_width=True, height=400)
        
        # Estadísticas rápidas de conexiones
        if not df_conexiones_por_capitulo.empty:
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                avg_connections = df_conexiones_por_capitulo['numero_conexiones'].mean()
                st.metric("Promedio conexiones/capítulo", f"{avg_connections:.1f}")
            with col_stats2:
                max_connections = df_conexiones_por_capitulo['numero_conexiones'].max()
                st.metric("Máximo conexiones", max_connections)
            with col_stats3:
                min_connections = df_conexiones_por_capitulo['numero_conexiones'].min()
                st.metric("Mínimo conexiones", min_connections)
        
        # Descargar conexiones por capítulo
        if not df_conexiones_por_capitulo.empty:
            filename_parts = ["conexiones_por_capitulo_keywords", f"p{percentil}", f"r{threshold:.2f}", similarity_method]
            
            st.download_button(
                "📥 Descargar Conexiones por Capítulo (CSV)",
                df_conexiones_por_capitulo.to_csv(index=False),
                "_".join(filename_parts) + ".csv", "text/csv",
                key="download_conexiones_por_capitulo_keywords_unique"
            )
        
        # --- ANÁLISIS GRÁFICO ---
        st.subheader("📊 Análisis de las Conexiones")
        
        col_anal1, col_anal2 = st.columns(2)
        
        with col_anal1:
            # Gráfico de distribución por curso con filtro
            st.write("**🎯 Filtro de Conexiones**")
            filtrar_mayor_grado = st.checkbox(
                "Mostrar solo conexiones con cursos de mayor grado", False,
                help="Si está activado, solo se considerarán conexiones donde al menos un curso sea de grado superior",
                key="filtrar_mayor_grado_keywords"
            )
            
            connections_list_filtrada = filtrar_conexiones_mayor_grado(connections_list) if filtrar_mayor_grado else connections_list
            titulo_grafico = "📚 Conexiones con Cursos de Mayor Grado" if filtrar_mayor_grado else "📚 Distribución de Conexiones por Curso"
            
            if connections_list_filtrada:
                course_connections = []
                for conn in connections_list_filtrada:
                    course_connections.extend([conn['curso_capitulo_1'], conn['curso_capitulo_2']])
                
                course_counts = pd.Series(course_connections).value_counts().reset_index()
                course_counts.columns = ['Curso', 'Conexiones']
                
                fig_courses = px.bar(course_counts, x='Curso', y='Conexiones', title=titulo_grafico,
                                   color='Conexiones', color_continuous_scale='viridis', text='Conexiones')
                fig_courses.update_layout(xaxis_tickangle=45, plot_bgcolor='rgba(248,249,250,1)',
                                        paper_bgcolor='rgba(248,249,250,1)', showlegend=False, height=400)
                fig_courses.update_traces(textposition='outside', marker_line_color='black', marker_line_width=1)
                st.plotly_chart(fig_courses, use_container_width=True, key="fig_courses_3")
                
                if filtrar_mayor_grado:
                    st.info(f"**Conexiones con mayor grado:** {len(connections_list_filtrada)} de {len(connections_list)} totales ({(len(connections_list_filtrada)/len(connections_list)*100):.1f}%)")
            else:
                st.warning("No hay conexiones que cumplan con el filtro seleccionado.")
        
        with col_anal2:
            # Histograma de similitudes
            if connections_list:
                sim_values = [conn['similitud_original'] for conn in connections_list]
                titulo_hist = f"📊 Distribución de {'Distancias' if similarity_method == 'euclidean' else 'Similitudes'}"
                
                fig_hist = px.histogram(x=sim_values, title=titulo_hist, nbins=20,
                                      labels={'x': f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}", 'y': 'Frecuencia'},
                                      color_discrete_sequence=['#1f77b4'], opacity=0.8, marginal="box")
                
                fig_hist.add_vline(x=threshold, line_dash="dot", line_color="blue",
                                 annotation_text=f"Umbral: {threshold:.2f}")
                
                fig_hist.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)',
                                     height=400, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist_3")
        
        # --- ANÁLISIS DE TIPOS DE CONEXIÓN ---
        st.subheader("🔗 Análisis de Tipos de Conexión")
        
        col_anal3, col_anal4 = st.columns(2)
        
        with col_anal3:
            # Gráfico de pie
            if connections_list:
                intra_curso = len([c for c in connections_list if c['curso_capitulo_1'] == c['curso_capitulo_2']])
                inter_curso = len(connections_list) - intra_curso
                
                fig_pie = px.pie(values=[intra_curso, inter_curso], names=['Intra-curso', 'Inter-curso'],
                               title="🎯 Tipos de Conexiones", color=['Intra-curso', 'Inter-curso'],
                               color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'})
                fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0],
                                    marker=dict(line=dict(color='white', width=2)))
                fig_pie.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)',
                                    height=350, showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie_3")
        
        with col_anal4:
            # Violin plot
            if connections_list:
                scatter_data = [{
                    'similitud': conn['similitud_original'],
                    'tipo': 'Intra-curso' if conn['curso_capitulo_1'] == conn['curso_capitulo_2'] else 'Inter-curso'
                } for conn in connections_list]
                
                df_scatter = pd.DataFrame(scatter_data)
                fig_violin = px.violin(df_scatter, x='tipo', y='similitud', color='tipo', box=True, points="all",
                                     title="🎻 Distribución por Tipo de Conexión",
                                     labels={'similitud': f"{'Distancia' if similarity_method == 'euclidean' else 'Similitud'}"},
                                     color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'})
                
                fig_violin.add_hline(y=threshold, line_dash="dot", line_color="blue", line_width=2,
                                   annotation_text=f"Umbral: {threshold:.4f}")
                
                fig_violin.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)',
                                       height=350, showlegend=False)
                st.plotly_chart(fig_violin, use_container_width=True, key="fig_violin_3")
        
        # --- ESTADÍSTICAS RÁPIDAS ---
        st.subheader("📈 Estadísticas Rápidas")

        # CALCULAR densidad aquí antes de usarla
        total_possible_connections = len(similarity_matrix) * (len(similarity_matrix) - 1) / 2
        actual_connections = np.sum(adjacency_matrix) / 2

        # Calcular densidad de forma segura
        try:
            if total_possible_connections > 0:
                densidad = float(actual_connections) / float(total_possible_connections)
            else:
                densidad = 0.0
            # Asegurar que no sea NaN y esté en rango [0,1]
            if np.isnan(densidad):
                densidad = 0.0
            densidad = max(0.0, min(1.0, densidad))
        except (ZeroDivisionError, ValueError, TypeError) as e:
            st.warning(f"Error calculando densidad: {e}")
            densidad = 0.0

        # Ahora sí usar densidad en las estadísticas
        sim_values = [conn['similitud_original'] for conn in connections_list] if connections_list else []
        intra_curso = len([c for c in connections_list if c['curso_capitulo_1'] == c['curso_capitulo_2']]) if connections_list else 0
        inter_curso = len(connections_list) - intra_curso if connections_list else 0

        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

        # Métricas con manejo de errores
        with col_stats1: 
            st.metric("📊 Conexiones Totales", len(connections_list), delta=f"{intra_curso} intra-curso")

        with col_stats2: 
            inter_percent = (inter_curso/len(connections_list)*100) if connections_list and len(connections_list) > 0 else 0
            st.metric("🔗 Conexiones Inter-curso", inter_curso, delta=f"{inter_percent:.1f}%")

        with col_stats3: 
            sim_mean = np.mean(sim_values) if sim_values else 0
            sim_std = np.std(sim_values) if sim_values else 0
            st.metric(f"📏 {'Distancia' if similarity_method == 'euclidean' else 'Similitud'} Promedio", 
                    f"{sim_mean:.3f}", delta=f"±{sim_std:.3f}")

        with col_stats4: 
            # Asegurar que densidad sea un número válido antes de formatear
            try:
                densidad_val = float(densidad)
                densidad_formatted = f"{densidad_val:.3f}"
                delta_percent = f"{(densidad_val*100):.1f}%"
            except (ValueError, TypeError):
                densidad_formatted = "0.000"
                delta_percent = "0.0%"
            
            st.metric("🎯 Densidad de Red", densidad_formatted, delta=delta_percent)

    else:
        st.warning("No se encontraron conexiones que cumplan con el umbral seleccionado. Intenta ajustar el umbral.")
            
    # --- MATRIZ COMPLETA Y ESTADÍSTICAS FINALES ---
    st.subheader("🔥 Matriz de Similitud Filtrada por Percentil")
    
    try:
        fig_matrix = create_similarity_heatmap(similarity_matrix, labels, 
                                             f"Matriz de Similitud Filtrada (p={percentil}%) ({method_names[similarity_method]})", 
                                             similarity_method)
        if fig_matrix: st.plotly_chart(fig_matrix, use_container_width=True, key="fig_matrix_2")
    except Exception as e:
        st.error(f"Error creando matriz de similitud filtrada: {str(e)}")
    
    # Estadísticas finales
    st.subheader("📊 Estadísticas de la Matriz de Similitud Filtrada")
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    col1, col2, col_stats3, col_stats4 = st.columns(4)
    if similarity_method == 'euclidean':
        with col1: st.metric("Distancia Promedio", f"{np.mean(upper_triangle):.3f}")
        with col2: st.metric("Distancia Máxima", f"{np.max(upper_triangle):.3f}")
        with col_stats3: st.metric("Distancia Mínima", f"{np.min(upper_triangle):.3f}")
        with col_stats4: st.metric("Desviación Estándar", f"{np.std(upper_triangle):.3f}")
    else:
        with col1: st.metric("Similitud Promedio", f"{np.mean(upper_triangle):.3f}")
        with col2: st.metric("Similitud Máxima", f"{np.max(upper_triangle):.3f}")
        with col_stats3: st.metric("Similitud Mínima", f"{np.min(upper_triangle):.3f}")
        with col_stats4: st.metric("Desviación Estándar", f"{np.std(upper_triangle):.3f}")
    
    # --- DESCARGAS FINALES ---
    st.subheader("📥 Descargas Finales")
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if st.button("💾 Descargar Matriz de Similitud como CSV", key="download_matrix_btn_keywords"):
            matrix_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
            st.download_button("Descargar Matriz CSV", matrix_df.to_csv(), 
                             f"matriz_similitud_keywords_p{percentil}_{similarity_method}.csv", "text/csv",
                             key="download_matrix_keywords")
    
    with col_dl2:
        if st.button("📊 Descargar Estadísticas Completas", key="download_stats_btn_keywords"):
            stats_data = {
                'Método': [similarity_method], 'Capítulos Analizados': [len(df_unique)],
                'Conexiones Totales': [len(connections_list) if connections_list else 0],
                'Densidad Red': [densidad], 'Percentil Aplicado': [percentil],
                'Umbral Absoluto': [threshold],
                'Similitud Promedio': [np.mean(upper_triangle)], 'Similitud Máxima': [np.max(upper_triangle)],
                'Similitud Mínima': [np.min(upper_triangle)], 'Desviación Estándar': [np.std(upper_triangle)]
            }
            stats_df = pd.DataFrame(stats_data)
            st.download_button("Descargar Estadísticas CSV", stats_df.to_csv(index=False),
                             f"estadisticas_keywords_p{percentil}_{similarity_method}.csv", "text/csv",
                             key="download_stats_keywords")

# Extra tab with percentile methodology for keywords - VERSION "CORREGIDA" uso de dataframe extra, creo que esto aplica otro tipo de método
def similarity_heatmaps_tab_with_percentiles_keywords_dataframe_esp():
    """Pestaña para heatmaps de similitud con percentiles en las keywords y umbrales"""
    st.markdown('<h1 class="main-header">🎯 Método de Percentiles por Keywords</h1>', unsafe_allow_html=True)
    
    # Cargar datos principales
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el archivo CSV esté en la carpeta 'data/'.")
        return
    
    # --- CONFIGURACIÓN ---
    st.subheader("⚙ Configuración")
    
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
            key="similarity_percentiles_all_metrics_corrected"
        )
    
    cursos_disponibles = ['Todos'] + ordenar_cursos_personalizado(df['curso'].unique().tolist())
    with col_sel2:
        curso_seleccionado = st.selectbox(
            "Filtrar por Curso:", cursos_disponibles,
            key="sim_percentiles_curso_all_metrics_corrected"
        )
    
    # --- PARÁMETROS ---
    st.subheader("🎯 Parámetros de la Metodología")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.markdown("**📊 Percentil (p)**")
        percentil = st.slider(
            "Percentil p:", 
            min_value=0, max_value=100, value=80, step=1,
            help="Conserva solo el top (100-p)% de las comparaciones más similares entre keywords",
            key="percentil_slider_all_metrics_corrected"
        )
        st.info(f"**p = {percentil}%**")
    
    with col_config2:
        st.markdown("**🎯 Umbral (r)**")
        if similarity_method == 'euclidean':
            threshold = st.slider(
                "Umbral r:", 
                min_value=0.0, max_value=2.0, value=1.0, step=0.01,
                help="Capítulos relacionados si distancia promedio ≤ r",
                key="threshold_slider_all_metrics_corrected"
            )
        else:
            threshold = st.slider(
                "Umbral r:", 
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                help="Capítulos relacionados si similitud promedio ≥ r",
                key="threshold_slider_all_metrics_corrected"
            )
        st.info(f"**r = {threshold:.2f}**")
    
    # --- FUNCIONES OPTIMIZADAS CORREGIDAS ---
    def crear_capitulo_id(row):
        """Crear identificador único de capítulo"""
        curso_abbr = str(row['curso']).replace('Primero', '1B').replace('Segundo', '2B')\
                                    .replace('Tercero', '3B').replace('Cuarto', '4B')\
                                    .replace('Quinto', '5B').replace('Sexto', '6B')
        return f"{curso_abbr}: Capítulo N°{row['numero']}: {row['titulo']}"
    
    def cargar_todas_metricas_precomputadas():
        """Cargar TODAS las métricas precomputadas desde archivo"""
        try:
            df_metrics = pd.read_csv("data/precomputed_all_keyword_metrics.csv")
            
            # Validar que el archivo tiene las columnas esperadas
            required_columns = ['capitulo_i', 'capitulo_j', 'cosine_similarities', 'euclidean_distances', 'dot_products']
            missing_columns = [col for col in required_columns if col not in df_metrics.columns]
            if missing_columns:
                st.error(f"❌ El archivo precomputado no tiene las columnas requeridas: {missing_columns}")
                return None
            
            # Convertir las columnas de métricas de string a lista
            def parse_metricas(metric_str):
                if isinstance(metric_str, str):
                    try:
                        # Manejar diferentes formatos de lista
                        metric_str = metric_str.strip()
                        if metric_str.startswith('[') and metric_str.endswith(']'):
                            metric_str = metric_str[1:-1]
                        # Filtrar valores vacíos y convertir a float
                        values = [float(x.strip()) for x in metric_str.split(',') if x.strip() and x.strip() != '...']
                        return values
                    except Exception as e:
                        st.warning(f"⚠️ Error parseando métricas: {e}")
                        return []
                return metric_str if isinstance(metric_str, list) else []
            
            df_metrics['cosine_similarities'] = df_metrics['cosine_similarities'].apply(parse_metricas)
            df_metrics['euclidean_distances'] = df_metrics['euclidean_distances'].apply(parse_metricas)
            df_metrics['dot_products'] = df_metrics['dot_products'].apply(parse_metricas)
            
            # CORRECCIÓN: Filtrar filas con métricas válidas (usar & con paréntesis)
            df_metrics = df_metrics[
                (df_metrics['cosine_similarities'].apply(len) > 0) & 
                (df_metrics['euclidean_distances'].apply(len) > 0) & 
                (df_metrics['dot_products'].apply(len) > 0)
            ]
            
            # Mostrar estadísticas de carga
            st.write(f"📊 Métricas cargadas: {len(df_metrics)} pares válidos")
            if len(df_metrics) > 0:
                sample_row = df_metrics.iloc[0]
                st.write(f"🔍 Ejemplo - Cosine: {len(sample_row['cosine_similarities'])} valores")
                st.write(f"🔍 Ejemplo - Euclidean: {len(sample_row['euclidean_distances'])} valores")
                st.write(f"🔍 Ejemplo - Dot: {len(sample_row['dot_products'])} valores")
            
            return df_metrics
        except FileNotFoundError:
            st.error("❌ No se encontró el archivo de métricas precomputadas.")
            st.info("💡 Ejecuta primero: `python precompute_all_keyword_similarities.py`")
            return None
        except Exception as e:
            st.error(f"❌ Error cargando métricas precomputadas: {e}")
            import traceback
            st.write("Detalles del error:")
            st.code(traceback.format_exc())
            return None
    
    def calcular_promedio_filtrado(metricas_ordenadas, percentil, method='cosine'):
        """
        Calcular promedio filtrado aplicando percentil a vector ordenado
        según el método específico
        """
        if not metricas_ordenadas or len(metricas_ordenadas) == 0:
            return 0.0, 0
        
        # CORRECCIÓN: Calcular índice de corte para el percentil (usar 100.0 para división float)
        n = len(metricas_ordenadas)
        indice_corte = max(1, int(n * (100 - percentil) / 100.0))
        
        if method == 'euclidean':
            # Para distancias: tomar las primeras (menores distancias) - ya están ordenadas de menor a mayor
            metricas_filtradas = metricas_ordenadas[:indice_corte]
        else:
            # Para similitudes: tomar las primeras (mayores valores) - ya están ordenadas de mayor a menor
            metricas_filtradas = metricas_ordenadas[:indice_corte]
        
        if len(metricas_filtradas) > 0:
            promedio = np.mean(metricas_filtradas)
            return promedio, len(metricas_filtradas)
        else:
            return 0.0, 0
    
    def obtener_metricas_por_metodo(df_metrics, method):
        """Obtener la columna correcta de métricas según el método seleccionado"""
        if method == 'cosine':
            return 'cosine_similarities'
        elif method == 'euclidean':
            return 'euclidean_distances'
        elif method == 'dot_product':
            return 'dot_products'
        else:
            return 'cosine_similarities'  # Default
    
    # --- PROCESAMIENTO PRINCIPAL ---
    df_filtrado = df if curso_seleccionado == 'Todos' else df[df['curso'] == curso_seleccionado]
    if len(df_filtrado) == 0:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return
    
    df_unique = df_filtrado.drop_duplicates(subset=['id']).copy()
    df_unique['capitulo_id'] = df_unique.apply(crear_capitulo_id, axis=1)
    
    labels = df_unique['capitulo_id'].tolist()
    courses = df_unique['curso'].tolist()
    
    # Cargar TODAS las métricas precomputadas
    with st.spinner("Cargando todas las métricas precomputadas..."):
        df_metrics = cargar_todas_metricas_precomputadas()
    
    if df_metrics is None or len(df_metrics) == 0:
        st.error("No se pudieron cargar las métricas precomputadas o están vacías.")
        return
    
    st.success(f"✅ Métricas precomputadas cargadas: {len(df_metrics)} pares de capítulos")
    
    # Filtrar métricas por curso seleccionado
    if curso_seleccionado != 'Todos':
        capitulos_filtrados = set(df_unique['id'].astype(int))
        
        df_metrics_filtrado = df_metrics[
            (df_metrics['capitulo_i'].astype(int).isin(capitulos_filtrados)) & 
            (df_metrics['capitulo_j'].astype(int).isin(capitulos_filtrados))
        ].copy()
    else:
        df_metrics_filtrado = df_metrics.copy()
    
    st.info(f"📊 Después del filtrado: {len(df_metrics_filtrado)} pares")
    
    if len(df_metrics_filtrado) == 0:
        st.warning("No hay pares de capítulos que coincidan con el filtro seleccionado.")
        return
    
    # Obtener columna de métricas según método seleccionado
    columna_metricas = obtener_metricas_por_metodo(df_metrics, similarity_method)
    
    # Calcular matriz aplicando percentil
    st.subheader(f"🧮 Aplicando Percentil a Métricas {similarity_method}")
    
    n_capitulos = len(df_unique)
    similarity_matrix = np.zeros((n_capitulos, n_capitulos))
    
    # Mapeo de ID de capítulo a índice
    id_to_idx = {}
    for idx, (_, row) in enumerate(df_unique.iterrows()):
        id_to_idx[int(row['id'])] = idx
    
    # Verificar mapeo
    st.write(f"🔍 Mapeo creado: {len(id_to_idx)} capítulos únicos")
    
    # Aplicar percentil a cada par
    progress_bar = st.progress(0)
    total_pares = len(df_metrics_filtrado)
    
    pares_procesados = 0
    pares_con_error = 0
    errores_detallados = []
    
    for idx, row in enumerate(df_metrics_filtrado.itertuples()):
        try:
            cap_id_i = int(row.capitulo_i)
            cap_id_j = int(row.capitulo_j)
            
            # Verificar que ambos IDs existen en el mapeo
            if cap_id_i not in id_to_idx or cap_id_j not in id_to_idx:
                pares_con_error += 1
                errores_detallados.append(f"ID no encontrado: {cap_id_i}-{cap_id_j}")
                continue
                
            metricas_ordenadas = getattr(row, columna_metricas)
            
            # Validar que las métricas no estén vacías
            if not metricas_ordenadas or len(metricas_ordenadas) == 0:
                pares_con_error += 1
                errores_detallados.append(f"Métricas vacías: {cap_id_i}-{cap_id_j}")
                continue
            
            idx_i = id_to_idx[cap_id_i]
            idx_j = id_to_idx[cap_id_j]
            
            # Calcular promedio filtrado
            promedio_filtrado, keywords_consideradas = calcular_promedio_filtrado(
                metricas_ordenadas, percentil, similarity_method
            )
            
            similarity_matrix[idx_i, idx_j] = promedio_filtrado
            similarity_matrix[idx_j, idx_i] = promedio_filtrado
            
            pares_procesados += 1
            
            if pares_procesados % max(1, total_pares // 10) == 0:
                progress_bar.progress(min(1.0, pares_procesados / total_pares))
                
        except Exception as e:
            pares_con_error += 1
            errores_detallados.append(f"Error procesando {cap_id_i}-{cap_id_j}: {str(e)}")
            continue
    
    progress_bar.empty()
    
    if pares_con_error > 0:
        st.warning(f"⚠️ {pares_con_error} pares tuvieron problemas")
        with st.expander("Ver detalles de errores"):
            for error in errores_detallados[:10]:  # Mostrar solo primeros 10
                st.write(f"• {error}")
    
    st.success(f"✅ {pares_procesados} pares procesados correctamente")
    
    # Completar diagonal según método
    for i in range(n_capitulos):
        if similarity_method == 'euclidean':
            similarity_matrix[i, i] = 0.0  # Distancia a sí mismo = 0
        else:
            similarity_matrix[i, i] = 1.0  # Similitud consigo mismo = 1
    
    # --- RESULTADOS ---
    st.subheader("📊 Resultados Finales")
    
    # Aplicar umbral (CORRECCIÓN: manejar euclidean correctamente)
    if similarity_method == 'euclidean':
        # Para distancias: valores más bajos = más similares
        adjacency_matrix = (similarity_matrix <= threshold).astype(int)
    else:
        # Para similitudes: valores más altos = más similares  
        adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Estadísticas
    total_possible = n_capitulos * (n_capitulos - 1) // 2
    actual_connections = np.sum(adjacency_matrix) // 2
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("Conexiones", actual_connections)
    with col_stats2:
        porcentaje = (actual_connections / total_possible * 100) if total_possible > 0 else 0
        st.metric("Porcentaje", f"{porcentaje:.1f}%")
    with col_stats3:
        densidad = actual_connections / total_possible if total_possible > 0 else 0
        st.metric("Densidad", f"{densidad:.4f}")
    with col_stats4:
        # Comparación con método tradicional (opcional)
        try:
            embeddings_matrix_trad = np.vstack(df_unique['embeddings_array'].values)
            similarity_trad = calculate_similarity_matrix(embeddings_matrix_trad, similarity_method)
            if similarity_method == 'euclidean':
                adj_trad = (similarity_trad <= threshold).astype(int)
            else:
                adj_trad = (similarity_trad >= threshold).astype(int)
            np.fill_diagonal(adj_trad, 0)
            conexiones_trad = np.sum(adj_trad) // 2
            diferencia = actual_connections - conexiones_trad
            st.metric("Δ vs Tradicional", f"{diferencia:+d}")
        except:
            st.metric("Δ vs Tradicional", "N/A")
    
    
    # --- VISUALIZACIONES ---  
    st.subheader(f"🔥 Matriz {similarity_method} Filtrada")
    title_sim = f"S_ij^(p) - {similarity_method} - p={percentil}%"
    fig_sim = create_similarity_heatmap(similarity_matrix, labels, title_sim, similarity_method)
    if fig_sim:
        st.plotly_chart(fig_sim, use_container_width=True, key="fig_sim_1")
    

    st.subheader("🎯 Matriz de Adyacencia")
    title_adj = f"Adyacencia - {similarity_method} - p={percentil}%, r={threshold:.2f}"
    fig_adj = create_similarity_heatmap(adjacency_matrix, labels, title_adj, 'binary')
    if fig_adj:
        st.plotly_chart(fig_adj, use_container_width=True, key="fig_adj_1")
    
    # --- ANÁLISIS DETALLADO ---
    st.subheader("🔍 Análisis Detallado")

    # Crear lista de conexiones
    connections_list = []
    for i in range(n_capitulos):
        for j in range(i + 1, n_capitulos):
            if adjacency_matrix[i, j] == 1:
                # Buscar información del par en las métricas precomputadas
                cap_id_i = df_unique.iloc[i]['id']
                cap_id_j = df_unique.iloc[j]['id']
                
                # Buscar el par en las métricas precomputadas
                par_info = None
                for row in df_metrics_filtrado.itertuples():
                    if (int(row.capitulo_i) == cap_id_i and int(row.capitulo_j) == cap_id_j) or \
                    (int(row.capitulo_i) == cap_id_j and int(row.capitulo_j) == cap_id_i):
                        par_info = row
                        break
                
                if par_info:
                    # Calcular porcentaje filtrado basado en el percentil aplicado
                    metricas_ordenadas = getattr(par_info, columna_metricas)
                    n_total = len(metricas_ordenadas)
                    indice_corte = max(1, int(n_total * (100 - percentil) / 100))
                    n_consideradas = indice_corte
                    porcentaje_filtrado = (n_consideradas / n_total * 100) if n_total > 0 else 0
                    
                    connections_list.append({
                        'curso_1': courses[i],
                        'capitulo_1': labels[i],
                        'curso_2': courses[j],
                        'capitulo_2': labels[j],
                        'similitud_filtrada': similarity_matrix[i, j],
                        'keywords_consideradas': n_consideradas,
                        'keywords_totales': n_total,
                        'porcentaje_filtrado': f"{porcentaje_filtrado:.1f}%",
                        'num_keywords_i': par_info.num_keywords_i,
                        'num_keywords_j': par_info.num_keywords_j
                    })
                else:
                    # Si no encontramos información del par, usar valores por defecto
                    connections_list.append({
                        'curso_1': courses[i],
                        'capitulo_1': labels[i],
                        'curso_2': courses[j],
                        'capitulo_2': labels[j],
                        'similitud_filtrada': similarity_matrix[i, j],
                        'keywords_consideradas': 'N/A',
                        'keywords_totales': 'N/A',
                        'porcentaje_filtrado': 'N/A',
                        'num_keywords_i': 'N/A',
                        'num_keywords_j': 'N/A'
                    })

    if connections_list:
        connections_df = pd.DataFrame(connections_list)
        
        # Estadísticas de similitudes (solo valores numéricos)
        sim_values_numeric = [conn['similitud_filtrada'] for conn in connections_list if isinstance(conn['similitud_filtrada'], (int, float))]
        
        if sim_values_numeric:
            col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
            with col_sim1:
                st.metric("Media S_ij", f"{np.mean(sim_values_numeric):.4f}")
            with col_sim2:
                st.metric("Mediana S_ij", f"{np.median(sim_values_numeric):.4f}")
            with col_sim3:
                st.metric("Desv. S_ij", f"{np.std(sim_values_numeric):.4f}")
            with col_sim4:
                st.metric("Rango S_ij", f"{np.ptp(sim_values_numeric):.4f}")
        else:
            st.warning("No hay valores numéricos para calcular estadísticas")
        
        # Mostrar estadísticas adicionales de keywords
        st.subheader("📊 Estadísticas de Keywords")
        
        # Calcular estadísticas de keywords consideradas (solo para pares con información)
        pares_con_info = [conn for conn in connections_list if conn['keywords_totales'] != 'N/A']
        
        if pares_con_info:
            total_keywords_consideradas = sum([conn['keywords_consideradas'] for conn in pares_con_info])
            total_keywords_totales = sum([conn['keywords_totales'] for conn in pares_con_info])
            porcentaje_promedio_filtrado = (total_keywords_consideradas / total_keywords_totales * 100) if total_keywords_totales > 0 else 0
            
            col_kw1, col_kw2, col_kw3 = st.columns(3)
            with col_kw1:
                st.metric("Keywords Consideradas", f"{total_keywords_consideradas:,}")
            with col_kw2:
                st.metric("Keywords Totales", f"{total_keywords_totales:,}")
            with col_kw3:
                st.metric("% Filtrado Promedio", f"{porcentaje_promedio_filtrado:.1f}%")
        
        # Mostrar conexiones
        st.subheader("🔗 Conexiones Identificadas")
        st.write(f"**Total de conexiones encontradas: {len(connections_list)}**")
        
        # Crear DataFrame para mostrar (columnas seleccionadas)
        display_columns = ['curso_1', 'capitulo_1', 'curso_2', 'capitulo_2', 'similitud_filtrada']
        if any(conn['keywords_consideradas'] != 'N/A' for conn in connections_list):
            display_columns.extend(['keywords_consideradas', 'keywords_totales', 'porcentaje_filtrado'])
        
        display_df = connections_df[display_columns].copy()
        
        # Renombrar columnas para mejor visualización
        column_rename = {
            'curso_1': 'Curso 1',
            'capitulo_1': 'Capítulo 1', 
            'curso_2': 'Curso 2',
            'capitulo_2': 'Capítulo 2',
            'similitud_filtrada': 'Similitud Filtrada',
            'keywords_consideradas': 'Keywords Consideradas',
            'keywords_totales': 'Keywords Totales',
            'porcentaje_filtrado': '% Filtrado'
        }
        display_df = display_df.rename(columns=column_rename)
        
        st.dataframe(display_df.head(10), use_container_width=True)
        
        # Histograma de similitudes
        if sim_values_numeric:
            fig_hist = px.histogram(
                x=sim_values_numeric,
                title=f"Distribución de S_ij^(p) - p={percentil}%",
                labels={'x': 'Similitud Promedio Filtrada', 'y': 'Frecuencia'},
                nbins=20,
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                            annotation_text=f"Umbral r={threshold:.2f}")
            
            # Agregar estadísticas al gráfico
            fig_hist.add_annotation(
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                text=f"Media: {np.mean(sim_values_numeric):.3f}<br>Mediana: {np.median(sim_values_numeric):.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist_4")
        
        # --- ANÁLISIS ADICIONAL: CONEXIONES POR CURSO ---
        st.subheader("📚 Distribución por Curso")
        
        # Contar conexiones por curso
        curso_conteo = {}
        for conn in connections_list:
            curso1 = conn['curso_1']
            curso2 = conn['curso_2']
            
            curso_conteo[curso1] = curso_conteo.get(curso1, 0) + 1
            curso_conteo[curso2] = curso_conteo.get(curso2, 0) + 1
        
        if curso_conteo:
            df_curso_conteo = pd.DataFrame({
                'Curso': list(curso_conteo.keys()),
                'Conexiones': list(curso_conteo.values())
            }).sort_values('Conexiones', ascending=False)
            
            fig_cursos = px.bar(
                df_curso_conteo,
                x='Curso',
                y='Conexiones',
                title="Conexiones por Curso",
                color='Conexiones',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_cursos, use_container_width=True, key="fig_cursos_1")
        
        # --- TIPOS DE CONEXIÓN ---
        st.subheader("🔀 Tipos de Conexión")
        
        intra_curso = len([conn for conn in connections_list if conn['curso_1'] == conn['curso_2']])
        inter_curso = len(connections_list) - intra_curso
        
        col_tipo1, col_tipo2 = st.columns(2)
        with col_tipo1:
            fig_tipo = px.pie(
                values=[intra_curso, inter_curso],
                names=['Intra-curso', 'Inter-curso'],
                title="Distribución de Tipos de Conexión",
                color=['Intra-curso', 'Inter-curso'],
                color_discrete_map={'Intra-curso': '#FF6B6B', 'Inter-curso': '#4ECDC4'}
            )
            st.plotly_chart(fig_tipo, use_container_width=True, key="fig_tipo_1")
        
        with col_tipo2:
            st.metric("Conexiones Intra-curso", intra_curso)
            st.metric("Conexiones Inter-curso", inter_curso)
            st.metric("Proporción Intra/Inter", f"{(intra_curso/max(inter_curso,1)):.2f}")
        
        # --- DESCARGAS ---
        st.subheader("📥 Descargar Resultados")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            matrix_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
            csv_matrix = matrix_df.to_csv()
            st.download_button(
                label="💾 Matriz S_ij^(p)",
                data=csv_matrix,
                file_name=f"matriz_similitud_filtrada_p{percentil}.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            adj_df = pd.DataFrame(adjacency_matrix, index=labels, columns=labels)
            csv_adj = adj_df.to_csv()
            st.download_button(
                label="🎯 Matriz Adyacencia",
                data=csv_adj,
                file_name=f"matriz_adyacencia_p{percentil}_r{threshold:.2f}.csv",
                mime="text/csv"
            )
        
        with col_dl3:
            csv_conn = connections_df.to_csv(index=False)
            st.download_button(
                label="📊 Lista Conexiones",
                data=csv_conn,
                file_name=f"conexiones_p{percentil}_r{threshold:.2f}.csv",
                mime="text/csv"
            )
    else:
        st.warning("No se encontraron conexiones con los parámetros actuales.")

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
    tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs(["📊 Embeddings", "🔥 Similitud", "🧱 Umbrales de Similitud", "🎯 Percentiles", "🧮 Percentiles Keywords","🔍 Búsqueda Semántica"])
    
    with tab1:
        embeddings_tab()
    
    with tab2:
        similarity_heatmaps_tab()
    
    with tab3:
        similarity_heatmaps_tab_with_threshold()
    
    with tab4:
        similarity_heatmaps_tab_with_percentiles()
    
    with tab6:
        #similarity_heatmaps_tab_with_percentiles_keywords_dataframe_esp()
        similarity_heatmaps_tab_with_percentiles_keywords()

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