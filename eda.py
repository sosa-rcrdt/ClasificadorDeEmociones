import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configuración general

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATH = "hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet"

label_a_emocion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

emociones_orden = ["sadness", "joy", "love", "anger", "fear", "surprise"]

plt.style.use("ggplot")
sns.set_theme(font_scale=1.1)

# Funciones auxiliares

def limpiar_texto_para_frecuencias(texto: str):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return [palabra for palabra in texto.split() if len(palabra) > 3]

def revisar_ejemplos_conflictivos(df, label_a_emocion, output_dir, max_ejemplos=100, random_state=42):
    df_conflictos = (
        df.groupby("text")
        .agg(
            n_registros=("label", "size"),
            n_etiquetas_distintas=("label", "nunique"),
            etiquetas=("label", lambda x: sorted(set(x)))
        )
        .reset_index()
    )

    df_conflictos = df_conflictos[df_conflictos["n_etiquetas_distintas"] > 1].copy()

    if df_conflictos.empty:
        print("\nNo se encontraron textos conflictivos.")
        return None, None, None

    df_conflictos["emociones"] = df_conflictos["etiquetas"].apply(
        lambda etiquetas: [label_a_emocion[e] for e in etiquetas]
    )
    df_conflictos["combinacion_emociones"] = df_conflictos["emociones"].apply(
        lambda emociones: " | ".join(emociones)
    )

    df_resumen_conflictos = (
        df_conflictos["combinacion_emociones"]
        .value_counts()
        .rename_axis("combinacion_emociones")
        .to_frame("Cantidad de textos")
        .reset_index()
    )

    print("\nCombinaciones de emociones en textos conflictivos:")
    print(f"{'Combinacion de emociones':<35} {'Cantidad de textos':>20}")
    print("-" * 55)
    for _, row in df_resumen_conflictos.head(15).iterrows():
        comb = row['combinacion_emociones']
        cant = row['Cantidad de textos']
        print(f"{comb:<35} {cant:>20}")

    n_muestra = min(max_ejemplos, len(df_conflictos))
    df_muestra_conflictos = df_conflictos.sample(n=n_muestra, random_state=random_state).copy()

    textos_muestra = df_muestra_conflictos["text"].tolist()

    df_ejemplos_conflictivos = (
        df[df["text"].isin(textos_muestra)][["text", "label", "emocion"]]
        .sort_values(["text", "label"])
        .reset_index(drop=True)
    )

    return df_conflictos, df_resumen_conflictos, df_ejemplos_conflictivos

# Carga del dataset

df = pd.read_parquet(DATASET_PATH)
df["emocion"] = df["label"].map(label_a_emocion)

# 1. Inspección general

print("Vista previa:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos:")
print(df.isnull().sum())

print(f"\nEl dataset contiene {df.shape[0]} filas y {df.shape[1]} columnas.\n")

# 2. Distribución de clases

df_conteo_emociones = (
    df["emocion"]
    .value_counts(dropna=False)
    .reindex(emociones_orden)
    .rename_axis("emocion")
    .to_frame("Cantidad de registros")
)

df_proporciones_emociones = (
    df["emocion"]
    .value_counts(normalize=True, dropna=False)
    .reindex(emociones_orden)
    .mul(100)
    .round(2)
    .rename_axis("emocion")
    .to_frame("Porcentaje")
)

df_resumen_emociones = df_conteo_emociones.join(df_proporciones_emociones)

print("Cantidad total de registros por emoción:")
print(df_conteo_emociones)

print("\nProporción de registros por emoción (%):")
print(df_proporciones_emociones)

print("\nResumen de clases:")
print(df_resumen_emociones)

df_plot_clases = df_resumen_emociones.reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_plot_clases,
    x="emocion",
    y="Cantidad de registros",
    hue="emocion",
    order=emociones_orden,
    dodge=False,
    legend=False
)
ax.set_title("Distribución de registros por emoción", fontsize=16)
ax.set_xlabel("Emoción", fontsize=12)
ax.set_ylabel("Cantidad de registros", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "distribucion_emociones.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 3. Longitud de texto

df["num_caracteres"] = df["text"].astype(str).str.len()
df["num_palabras"] = df["text"].astype(str).str.split().str.len()

print("\nResumen global de longitud de texto:")

print("\n- Caracteres:")
print(df["num_caracteres"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n- Palabras:")
print(df["num_palabras"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

df_longitud_por_emocion = (
    df.groupby("emocion")[["num_caracteres", "num_palabras"]]
    .agg(["mean", "median", "min", "max"])
    .reindex(emociones_orden)
    .round(2)
)

print("\nResumen de longitud por emoción:")
print(df_longitud_por_emocion)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="emocion",
    y="num_palabras",
    order=emociones_orden
)
plt.title("Distribución de longitud (número de palabras) por emoción", fontsize=16)
plt.xlabel("Emoción", fontsize=12)
plt.ylabel("Número de palabras", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "boxplot_longitud_palabras_por_emocion.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 4. Textos muy cortos

df_textos_cortos = pd.DataFrame({
    "Criterio": [
        "1 palabra o menos",
        "2 palabras o menos",
        "3 palabras o menos",
        "5 palabras o menos"
    ],
    "Cantidad": [
        (df["num_palabras"] <= 1).sum(),
        (df["num_palabras"] <= 2).sum(),
        (df["num_palabras"] <= 3).sum(),
        (df["num_palabras"] <= 5).sum()
    ]
})

df_textos_cortos["Porcentaje"] = (
    df_textos_cortos["Cantidad"] / len(df) * 100
).round(2)

print("\nCantidad de textos muy cortos:")
print(df_textos_cortos.to_string(index=False))

df_textos_cortos_por_emocion = (
    df.assign(texto_muy_corto=df["num_palabras"] <= 3)
    .groupby("emocion")["texto_muy_corto"]
    .agg(["sum", "mean"])
    .rename(columns={"sum": "Cantidad", "mean": "Proporcion"})
    .reindex(emociones_orden)
)

df_textos_cortos_por_emocion["Proporcion"] = (
    df_textos_cortos_por_emocion["Proporcion"] * 100
).round(2)

print("\nTextos muy cortos (<= 3 palabras) por emoción:")
print(df_textos_cortos_por_emocion)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_textos_cortos_por_emocion.reset_index(),
    x="emocion",
    y="Proporcion",
    hue="emocion",
    order=emociones_orden,
    dodge=False,
    legend=False
)
ax.set_title("Proporción de textos muy cortos (<= 3 palabras) por emoción", fontsize=16)
ax.set_xlabel("Emoción", fontsize=12)
ax.set_ylabel("Porcentaje", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "textos_muy_cortos_por_emocion.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 5. Duplicados

total_duplicados_texto = df.duplicated(subset=["text"]).sum()
total_filas_en_grupos_duplicados = df.duplicated(subset=["text"], keep=False).sum()
total_duplicados_texto_label = df.duplicated(subset=["text", "label"]).sum()

df_conflicto_etiquetas = (
    df.groupby("text")["label"]
    .nunique()
    .reset_index(name="n_etiquetas_distintas")
)

df_conflicto_etiquetas = df_conflicto_etiquetas[
    df_conflicto_etiquetas["n_etiquetas_distintas"] > 1
]

print("\nDuplicados exactos de texto:")
print(f"- Filas duplicadas (sin contar la primera aparición): {total_duplicados_texto}")
print(f"- Filas que pertenecen a grupos de texto duplicado: {total_filas_en_grupos_duplicados}")
print(f"- Filas duplicadas exactas de texto + etiqueta: {total_duplicados_texto_label}")

print(f"\nTextos idénticos con etiquetas distintas: {len(df_conflicto_etiquetas)}")

df_resumen_duplicados = pd.DataFrame({
    "Metrica": [
        "Duplicados de texto (sin contar primera aparición)",
        "Filas dentro de grupos de texto duplicado",
        "Duplicados exactos de texto + etiqueta",
        "Textos idénticos con etiquetas distintas"
    ],
    "Valor": [
        total_duplicados_texto,
        total_filas_en_grupos_duplicados,
        total_duplicados_texto_label,
        len(df_conflicto_etiquetas)
    ]
})

print("\nResumen de duplicados:")
print(df_resumen_duplicados.to_string(index=False))

# 6. Revisión de ejemplos conflictivos

df_conflictos, df_resumen_conflictos, df_ejemplos_conflictivos = revisar_ejemplos_conflictivos(
    df=df,
    label_a_emocion=label_a_emocion,
    output_dir=OUTPUT_DIR,
    max_ejemplos=100,
    random_state=42
)

# 7. Nubes de palabras por emoción

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
        ).generate(texto_completo)

        axes[i].imshow(nube, interpolation="bilinear")
    else:
        axes[i].text(
            0.5,
            0.5,
            "No hay datos para esta emoción",
            ha="center",
            va="center",
            fontsize=12
        )

    axes[i].axis("off")
    axes[i].set_title(emocion, fontsize=16)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "wordclouds_por_emocion.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 8. Top 10 palabras frecuentes por emoción

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, emocion in enumerate(emociones_orden):
    textos = df.loc[df["emocion"] == emocion, "text"].dropna().astype(str).tolist()

    palabras = []
    for texto in textos:
        palabras.extend(limpiar_texto_para_frecuencias(texto))

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
plt.savefig(
    os.path.join(OUTPUT_DIR, "top10_palabras_por_emocion.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()