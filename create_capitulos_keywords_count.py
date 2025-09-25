import numpy as np
import pandas as pd

# Leer el archivo original
df = pd.read_csv("data/capitulos_keywords_with_embeddings.csv")

# Agrupar por id, curso, numero y titulo, y contar keywords
df_count = (
    df.groupby(["id", "curso", "numero", "titulo"])
    .size()
    .reset_index(name="num_keywords")
)

# Guardar a un nuevo archivo
df_count.to_csv("data/capitulos_keywords_count.csv", index=False)

print("Archivo generado: data/capitulos_keywords_count.csv")

