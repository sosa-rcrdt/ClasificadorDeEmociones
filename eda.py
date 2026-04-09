import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")

# Mostrar las primeras filas para inspección visual inicial
print("Vista previa:")
print(df.head())

# Ver tipos de datos por columna
print("\nTipos de datos:")
print(df.dtypes)

# Verificar valores nulos por columna
print("\nValores nulos:")
print(df.isnull().sum())

# Mostrar la forma del DataFrame (filas, columnas)
print(f"\nEl dataset contiene {df.shape[0]} filas y {df.shape[1]} columnas.\n")

label_a_emocion = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
}

# Conteo por emoción
df_conteo_emociones = (
    df["label"]
        .map(label_a_emocion)
        .value_counts(dropna=False)
        .rename_axis("emocion")
        .to_frame("Cantidad de registros")
)

print("\nCantidad total de registros por emoción:")
print(df_conteo_emociones)

# Estilos (opcional)
plt.style.use("ggplot")
sns.set_theme(font_scale=1.1)

# Preparar el DataFrame para graficar (tu índice es 'emocion')
df_plot = df_conteo_emociones.reset_index()  # columnas: ['emocion', 'Cantidad de registros']

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_plot,
    x="emocion",
    y="Cantidad de registros",
    hue="emocion",
    dodge=False,
    legend=False
)

ax.set_title("Distribución de registros por emoción", fontsize=16)
ax.set_xlabel("Emoción", fontsize=12)
ax.set_ylabel("Cantidad de registros", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Recomendado en scripts: guardar
plt.savefig(os.path.join(OUTPUT_DIR, "distribucion_emociones.png"), dpi=200, bbox_inches="tight")

# Asegura columna emocion (si no la tienes ya)
df["emocion"] = df["label"].map(label_a_emocion)

# Preparar figura 2x3 (6 emociones)
emociones_orden = ["sadness", "joy", "love", "anger", "fear", "surprise"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, emocion in enumerate(emociones_orden):
    textos = df.loc[df["emocion"] == emocion, "text"].dropna().astype(str).tolist()
    texto_completo = " ".join(textos)

    if texto_completo.strip():
        nube = WordCloud(
            width=800,
            height=400,
            background_color="white",
            collocations=False
            # 👇 SIN stopwords (como pediste)
        ).generate(texto_completo)

        axes[i].imshow(nube, interpolation="bilinear")
    else:
        axes[i].text(
            0.5, 0.5, "No hay datos para esta emoción",
            ha="center", va="center", fontsize=12
        )

    axes[i].axis("off")
    axes[i].set_title(emocion, fontsize=16)

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds_por_emocion.png"), dpi=200, bbox_inches="tight")

# Asegura columna emoción
df["emocion"] = df["label"].map(label_a_emocion)

def limpiar_texto_sin_stopwords(texto: str):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)   # solo letras inglesas
    texto = re.sub(r"\s+", " ", texto).strip()

    # 👇 Filtro mínimo 3 caracteres
    return [w for w in texto.split() if len(w) > 3]

emociones_orden = ["sadness", "joy", "love", "anger", "fear", "surprise"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, emocion in enumerate(emociones_orden):
    textos = df.loc[df["emocion"] == emocion, "text"].dropna().astype(str).tolist()

    palabras = []
    for t in textos:
        palabras.extend(limpiar_texto_sin_stopwords(t))

    top_palabras = Counter(palabras).most_common(10)
    palabras_df = pd.DataFrame(top_palabras, columns=["Palabra", "Frecuencia"])

    sns.barplot(
        data=palabras_df,
        x="Frecuencia",
        y="Palabra",
        hue="Palabra",
        dodge=False,
        legend=False,
        ax=axes[i]
    )

    axes[i].set_title(f"Top 10 palabras frecuentes - {emocion}", fontsize=14)
    axes[i].set_xlabel("Frecuencia")
    axes[i].set_ylabel("Palabra")

plt.tight_layout()

# Si guardas en outputs:
plt.savefig(os.path.join(OUTPUT_DIR, "top10_palabras_por_emocion.png"), dpi=200, bbox_inches="tight")
