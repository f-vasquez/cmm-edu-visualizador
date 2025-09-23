# 🎓 CMM-EDU Visualizador

Una aplicación web interactiva para la visualización y análisis de embeddings de contenido educativo. Desarrollada con Streamlit y técnicas de machine learning para el análisis de similitud semántica entre capítulos de cursos.

## ✨ Características Principales

- 📊 **Visualización de Embeddings**: Utiliza t-SNE y UMAP para reducir la dimensionalidad y visualizar los embeddings en 2D
- 🔥 **Matrices de Similitud**: Heatmaps inter-capítulo con múltiples métricas (Coseno, Euclidiana, Producto Punto)
- 🎯 **Filtros Interactivos**: Filtrado por curso y métodos de visualización personalizables
- 📈 **Análisis de Similitud**: Encuentra capítulos similares y rankings de similitud promedio
- 📋 **Dashboard de Análisis**: Estadísticas y métricas del contenido educativo
- 🔍 **Búsqueda Semántica**: Busca capítulos relevantes usando embeddings de OpenAI
- 🛠️ **Sistema de Pestañas**: Diseño extensible para futuras herramientas
- 🎨 **Interfaz Moderna**: UI responsive con tema personalizable

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.11 o superior
- pip o conda para gestión de paquetes

### Instalación Local

1. **Clonar el repositorio**
   ```bash
   git clone <tu-repositorio>
   cd cmm-edu-visualizador
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En macOS/Linux
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Preparar los datos**
   - Asegúrate de que tu archivo CSV esté en la carpeta `data/`
   - El archivo debe llamarse `capitulos_keywords_with_embeddings.csv`
   - Estructura requerida: `id`, `curso`, `numero`, `titulo`, `keywords`, `keywords_embedding`

5. **Configurar OpenAI (Opcional - solo para Búsqueda Semántica)**
   - Crea un archivo `.env` en la raíz del proyecto
   - Agrega tu API key de OpenAI:
     ```bash
     # Archivo .env
     OPENAI_API_KEY=sk-tu_api_key_completa_aqui
     ```
   - Obtén tu API key desde: https://platform.openai.com/api-keys
   - **Ejemplo completo del archivo .env**:
     ```
     # OpenAI API Key para búsqueda semántica
     OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yzabc567
     ```
   - **Nota**: Esta configuración solo es necesaria para la pestaña de "Búsqueda Semántica"
   - **Seguridad**: Nunca subas el archivo `.env` a repositorios públicos (ya está en .gitignore)

6. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```

6. **Abrir en el navegador**
   - La aplicación se abrirá automáticamente en `http://localhost:8501`

## 📂 Estructura del Proyecto

```
cmm-edu-visualizador/
├── app.py                 # Aplicación principal de Streamlit
├── requirements.txt       # Dependencias de Python
├── Dockerfile            # Configuración para contenedor Docker
├── railway.toml          # Configuración para Railway
├── .streamlit/
│   └── config.toml       # Configuración de Streamlit
├── data/
│   └── capitulos_keywords_with_embeddings.csv  # Datos de entrada
└── README.md             # Documentación
```

## 🔧 Configuración de Datos

### Formato del CSV

Tu archivo CSV debe tener las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | int | Identificador único del capítulo |
| `curso` | str | Nombre del curso (ej: "Primero Básico") |
| `numero` | int | Número del capítulo |
| `titulo` | str | Título del capítulo |
| `keywords` | str | Palabras clave del capítulo |
| `keywords_embedding` | str | Array de embeddings como string |

### Ejemplo de fila:
```csv
id,curso,numero,titulo,keywords,keywords_embedding
1,"Primero Básico",1,"Números hasta 10","• Números del 1 al 5","[0.1, -0.2, 0.3, ...]"
```

## 🚀 Deployment en Railway

Railway es una plataforma de deployment que facilita el despliegue de aplicaciones web.

### Pasos para el Deployment

1. **Crear cuenta en Railway**
   - Ve a [railway.app](https://railway.app)
   - Regístrate con tu cuenta de GitHub

2. **Conectar repositorio**
   - Haz push de tu código a GitHub
   - En Railway, crea un nuevo proyecto desde GitHub
   - Selecciona este repositorio

3. **Configuración automática**
   - Railway detectará automáticamente el `Dockerfile`
   - La configuración en `railway.toml` se aplicará automáticamente

4. **Variables de entorno**
   - En el dashboard de Railway, ve a la sección "Variables"
   - Agrega la variable `OPENAI_API_KEY` con tu API key de OpenAI
   - Esto habilitará la funcionalidad de búsqueda semántica en producción

5. **Deploy**
   - Railway iniciará el build automáticamente
   - Una vez completado, tendrás una URL pública para tu aplicación

### Configuración Manual de Railway

Si prefieres configurar manualmente:

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Inicializar proyecto
railway init

# Deploy
railway up
```

## 🎯 Uso de la Aplicación

### Pestaña de Embeddings 📊

- **Filtros**: Selecciona cursos específicos o visualiza todos
- **Métodos de Reducción**: Elige entre t-SNE y UMAP
- **Colorización**: Colorea puntos por curso o número de capítulo
- **Análisis de Similitud**: Encuentra capítulos con contenido similar

### Pestaña de Similitud 🔥

- **Matrices de Similitud**: Heatmaps completos 99x99 de similitud inter-capítulo
- **Múltiples Métricas**: Similitud Coseno, Distancia Euclidiana, Producto Punto
- **Similitud Promedio**: Visualización del promedio de similitud por capítulo
- **Rankings**: Top capítulos más similares y más únicos
- **Análisis de Pares**: Identificación de los pares más similares
- **Exportación**: Descarga matrices como CSV

### Pestaña de Análisis 📈

- **Distribución por Curso**: Gráficos de barras con estadísticas
- **Análisis de Keywords**: Palabras clave más frecuentes
- **Métricas Generales**: Estadísticas del dataset completo

### Pestaña de Búsqueda Semántica 🔍

- **Búsqueda por Texto**: Escribe cualquier consulta en lenguaje natural
- **Embedding con OpenAI**: Usa el modelo `text-embedding-3-small` para generar embeddings
- **Similitud Semántica**: Compara la consulta con las keywords de todos los capítulos
- **Optimización FAISS**: Búsquedas vectoriales ultra-rápidas con indexación eficiente
- **Visualización por Cursos**: Organiza resultados en 6 columnas (una por curso)
- **Sistema de Resaltado**: Los capítulos relevantes se destacan según el umbral de similitud
- **Top Rankings**: Muestra los 10 capítulos más relevantes para la consulta
- **Métricas de Rendimiento**: Timing detallado de embedding y búsqueda
- **Controles Interactivos**: Ajusta umbral de similitud y opciones de visualización
- **Exportación**: Descarga resultados de búsqueda como CSV

### Pestaña de Herramientas 🛠️

- Espacio reservado para futuras funcionalidades
- Ideas y roadmap de nuevas características

## 🔍 Características Técnicas

### Algoritmos de Reducción de Dimensionalidad

- **t-SNE**: Ideal para visualizar clusters locales y patrones
- **UMAP**: Preserva mejor la estructura global, más rápido en datasets grandes

### Métricas de Similitud

- **Similitud Coseno**: Para encontrar capítulos con contenido semánticamente similar
- **Normalización**: Los embeddings se normalizan para mejor comparación

### Optimizaciones

- **FAISS (Facebook AI Similarity Search)**: Biblioteca ultra-optimizada para búsquedas vectoriales
  - Índice `IndexFlatIP` para búsquedas exactas de productos internos
  - Normalización L2 para similitud coseno optimizada
  - Escalabilidad a millones de vectores con rendimiento constante
- **Caching**: Uso de `@st.cache_data` para optimizar carga de datos e índices
- **Lazy Loading**: Cálculos pesados solo cuando se necesitan
- **Responsive Design**: Interfaz adaptable a diferentes tamaños de pantalla

## 🛠️ Desarrollo

### Agregar Nuevas Funcionalidades

1. **Nueva pestaña**: Agrega función en `app.py` siguiendo el patrón de las existentes
2. **Nuevos análisis**: Extiende las funciones de análisis en las pestañas existentes
3. **Nuevas visualizaciones**: Usa Plotly para crear gráficos interactivos

### Estructura de Código

- `load_data()`: Carga y procesa el CSV
- `reduce_dimensions()`: Aplica t-SNE o UMAP
- `plot_embeddings()`: Crea visualizaciones con Plotly
- `*_tab()`: Funciones para cada pestaña

## 📋 Troubleshooting

### Problemas Comunes

**Error al cargar datos:**
- Verifica que el archivo CSV esté en `data/`
- Confirma que las columnas tengan los nombres correctos
- Asegúrate de que los embeddings sean arrays válidos

**Error de memoria en t-SNE:**
- Reduce el tamaño del dataset o usa UMAP
- Ajusta el parámetro `perplexity` en t-SNE

**Lentitud en la visualización:**
- Usa UMAP en lugar de t-SNE para datasets grandes
- Filtra por curso para reducir el número de puntos

### Logs y Debugging

```bash
# Ver logs de Streamlit
streamlit run app.py --logger.level=debug

# En Railway, revisa los logs en el dashboard
```

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Si tienes preguntas o problemas:

- Crea un issue en GitHub
- Revisa la documentación de [Streamlit](https://docs.streamlit.io/)
- Consulta la documentación de [Railway](https://docs.railway.app/)

---

**Desarrollado para CMM-EDU** | Visualización de Embeddings Educativos
