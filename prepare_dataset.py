import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuración general

DATASET_PATH = "hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet"
OUTPUT_DIR = "prepared_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TRAIN_SIZE = 0.80
VALID_SIZE = 0.10
TEST_SIZE = 0.10

label_a_emocion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

emociones_orden = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Funciones auxiliares

def imprimir_resumen_clases(df, nombre_split):
    conteo = (
        df["label"]
        .map(label_a_emocion)
        .value_counts(dropna=False)
        .reindex(emociones_orden)
        .fillna(0)
        .astype(int)
    )

    porcentaje = (
        df["label"]
        .map(label_a_emocion)
        .value_counts(normalize=True, dropna=False)
        .reindex(emociones_orden)
        .fillna(0)
        .mul(100)
        .round(2)
    )

    resumen = pd.DataFrame({
        "Cantidad": conteo,
        "Porcentaje": porcentaje
    })

    print(f"\nDistribución de clases en {nombre_split}:")
    print(resumen)

    return resumen


def elegir_etiqueta_estratificacion(labels_unicos, conteo_global_labels):
    if len(labels_unicos) == 1:
        return labels_unicos[0]

    return min(labels_unicos, key=lambda x: (conteo_global_labels[x], x))


def verificar_solapamiento_textos(train_df, val_df, test_df):
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    solap_train_val = len(train_texts.intersection(val_texts))
    solap_train_test = len(train_texts.intersection(test_texts))
    solap_val_test = len(val_texts.intersection(test_texts))

    return {
        "train_val": solap_train_val,
        "train_test": solap_train_test,
        "val_test": solap_val_test
    }

# Carga del dataset

df = pd.read_parquet(DATASET_PATH)
df["emocion"] = df["label"].map(label_a_emocion)

print("Forma original del dataset:")
print(df.shape)

print("\nDistribución original de clases:")
print(
    df["emocion"]
    .value_counts(dropna=False)
    .reindex(emociones_orden)
)

# Limpieza mínima

df_preparado = df.drop_duplicates(subset=["text", "label"]).copy()

print("\nForma después de eliminar duplicados exactos de texto + etiqueta:")
print(df_preparado.shape)

# Conteo global por etiqueta para ayudar en la estratificación de textos conflictivos

conteo_global_labels = df_preparado["label"].value_counts().to_dict()

# Construcción de grupos por texto

df_grupos = (
    df_preparado.groupby("text")
    .agg(
        n_registros=("label", "size"),
        labels_unicos=("label", lambda x: sorted(set(x)))
    )
    .reset_index()
)

df_grupos["n_labels_unicos"] = df_grupos["labels_unicos"].apply(len)
df_grupos["label_estratificacion"] = df_grupos["labels_unicos"].apply(
    lambda labels: elegir_etiqueta_estratificacion(labels, conteo_global_labels)
)
df_grupos["emocion_estratificacion"] = df_grupos["label_estratificacion"].map(label_a_emocion)

print("\nCantidad de textos únicos:")
print(len(df_grupos))

print("\nTextos con una sola etiqueta:")
print((df_grupos["n_labels_unicos"] == 1).sum())

print("\nTextos con múltiples etiquetas:")
print((df_grupos["n_labels_unicos"] > 1).sum())

# Split por texto
# Todos los registros del mismo texto quedarán en la misma partición.

df_train_texts, df_temp_texts = train_test_split(
    df_grupos,
    test_size=(1 - TRAIN_SIZE),
    random_state=RANDOM_STATE,
    stratify=df_grupos["label_estratificacion"]
)

# De lo restante, se divide mitad para validación y mitad para prueba.

df_val_texts, df_test_texts = train_test_split(
    df_temp_texts,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=df_temp_texts["label_estratificacion"]
)

# Reconstrucción de splits a nivel fila

train_texts = set(df_train_texts["text"])
val_texts = set(df_val_texts["text"])
test_texts = set(df_test_texts["text"])

train_df = df_preparado[df_preparado["text"].isin(train_texts)].copy()
val_df = df_preparado[df_preparado["text"].isin(val_texts)].copy()
test_df = df_preparado[df_preparado["text"].isin(test_texts)].copy()

# Columnas finales para entrenamiento

columnas_finales = ["text", "label"]
train_df = train_df[columnas_finales].reset_index(drop=True)
val_df = val_df[columnas_finales].reset_index(drop=True)
test_df = test_df[columnas_finales].reset_index(drop=True)

# Verificaciones

solapamientos = verificar_solapamiento_textos(train_df, val_df, test_df)

print("\nSolapamiento de textos entre particiones:")
print(solapamientos)

if any(valor > 0 for valor in solapamientos.values()):
    raise ValueError("Se detectó solapamiento de textos entre particiones.")

# Resúmenes

print("\nTamaños finales:")
print(f"Train: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test: {test_df.shape}")

resumen_train = imprimir_resumen_clases(train_df, "train")
resumen_val = imprimir_resumen_clases(val_df, "validation")
resumen_test = imprimir_resumen_clases(test_df, "test")

# Guardado de archivos principales

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding="utf-8")
val_df.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding="utf-8")

train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUTPUT_DIR, "validation.parquet"), index=False)
test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)

# Guardado de resúmenes

# resumen_train.to_csv(os.path.join(OUTPUT_DIR, "resumen_train.csv"), encoding="utf-8")
# resumen_val.to_csv(os.path.join(OUTPUT_DIR, "resumen_validation.csv"), encoding="utf-8")
# resumen_test.to_csv(os.path.join(OUTPUT_DIR, "resumen_test.csv"), encoding="utf-8")

# df_grupos.to_csv(
#     os.path.join(OUTPUT_DIR, "grupos_texto_para_split.csv"),
#     index=False,
#     encoding="utf-8"
# )

metadata = {
    "dataset_original_filas": int(df.shape[0]),
    "dataset_despues_deduplicacion_texto_label": int(df_preparado.shape[0]),
    "textos_unicos": int(len(df_grupos)),
    "textos_con_una_sola_etiqueta": int((df_grupos["n_labels_unicos"] == 1).sum()),
    "textos_con_multiples_etiquetas": int((df_grupos["n_labels_unicos"] > 1).sum()),
    "train_filas": int(train_df.shape[0]),
    "validation_filas": int(val_df.shape[0]),
    "test_filas": int(test_df.shape[0]),
    "train_textos_unicos": int(train_df["text"].nunique()),
    "validation_textos_unicos": int(val_df["text"].nunique()),
    "test_textos_unicos": int(test_df["text"].nunique()),
    "solapamiento_textos": solapamientos,
    "random_state": RANDOM_STATE,
    "train_size": TRAIN_SIZE,
    "validation_size": VALID_SIZE,
    "test_size": TEST_SIZE
}

# with open(os.path.join(OUTPUT_DIR, "metadata_preparacion.json"), "w", encoding="utf-8") as f:
#     json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"\nArchivos guardados en: {OUTPUT_DIR}")