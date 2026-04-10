import os
import json
import inspect

import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Configuración general

TRAIN_FILE = os.path.join("prepared_data", "train.csv")
VALIDATION_FILE = os.path.join("prepared_data", "validation.csv")

MODEL_CHECKPOINT = "FacebookAI/roberta-base"
OUTPUT_DIR = "roberta_finetuning"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

SEED = 42
MAX_LENGTH = 128
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_RATIO = 0.10
LOGGING_STEPS = 100
USE_CLASS_WEIGHTS = True
RESUME_FROM_CHECKPOINT = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
label2id = {v: k for k, v in id2label.items()}

set_seed(SEED)


# Funciones auxiliares

def guardar_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def construir_training_arguments():
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": LOGGING_STEPS,
        "save_total_limit": 2,
        "report_to": "none",
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "seed": SEED,
        "data_seed": SEED,
        "dataloader_num_workers": 0,
        "fp16": torch.cuda.is_available(),
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "save_safetensors" in signature:
        kwargs["save_safetensors"] = True

    return TrainingArguments(**kwargs)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# Carga de datos

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"No se encontró el archivo: {TRAIN_FILE}")

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(f"No se encontró el archivo: {VALIDATION_FILE}")

raw_datasets = load_dataset(
    "csv",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    }
)

raw_datasets["train"] = raw_datasets["train"].shuffle(seed=SEED).select(range(20000))
raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=SEED).select(range(4000))

raw_datasets = raw_datasets.rename_column("label", "labels")

print("Tamaño de los conjuntos:")
print(raw_datasets)

print("\nEjemplo de train:")
print(raw_datasets["train"][0])

print("\nEjemplo de validation:")
print(raw_datasets["validation"][0])

# Tokenizador

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_datasets = raw_datasets.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizando textos"
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# Pesos por clase

class_weights_tensor = None

if USE_CLASS_WEIGHTS:
    train_labels = np.array(raw_datasets["train"]["labels"])
    classes = np.array(sorted(id2label.keys()))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    print("\nPesos por clase:")
    for class_id, weight in zip(classes, class_weights):
        print(f"{class_id} ({id2label[int(class_id)]}): {weight:.6f}")

    guardar_json(
        {id2label[int(class_id)]: float(weight) for class_id, weight in zip(classes, class_weights)},
        os.path.join(OUTPUT_DIR, "class_weights.json")
    )

# Modelo

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

training_args = construir_training_arguments()

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_datasets["train"],
    "eval_dataset": tokenized_datasets["validation"],
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
    "class_weights": class_weights_tensor,
}

trainer_signature = inspect.signature(Trainer.__init__).parameters

if "processing_class" in trainer_signature:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = WeightedTrainer(**trainer_kwargs)

# Información de entrenamiento

config_entrenamiento = {
    "model_checkpoint": MODEL_CHECKPOINT,
    "max_length": MAX_LENGTH,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_ratio": WARMUP_RATIO,
    "use_class_weights": USE_CLASS_WEIGHTS,
    "seed": SEED,
    "train_file": TRAIN_FILE,
    "validation_file": VALIDATION_FILE,
}

guardar_json(config_entrenamiento, os.path.join(OUTPUT_DIR, "config_entrenamiento.json"))
guardar_json(
    {
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": label2id,
    },
    os.path.join(OUTPUT_DIR, "label_mapping.json")
)

# Entrenamiento

print("\nIniciando fine-tuning...\n")

train_result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

# Guardado final

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

train_metrics = train_result.metrics
eval_metrics = trainer.evaluate(tokenized_datasets["validation"])

trainer.state.save_to_json(os.path.join(OUTPUT_DIR, "trainer_state.json"))

guardar_json(train_metrics, os.path.join(OUTPUT_DIR, "train_metrics.json"))
guardar_json(eval_metrics, os.path.join(OUTPUT_DIR, "validation_metrics.json"))

best_model_info = {
    "best_metric": trainer.state.best_metric,
    "best_model_checkpoint": trainer.state.best_model_checkpoint,
    "final_model_dir": FINAL_MODEL_DIR,
}

guardar_json(best_model_info, os.path.join(OUTPUT_DIR, "best_model_info.json"))

print("\nEntrenamiento finalizado.")
print(f"Modelo final guardado en: {FINAL_MODEL_DIR}")
print(f"Métricas de train guardadas en: {os.path.join(OUTPUT_DIR, 'train_metrics.json')}")
print(f"Métricas de validation guardadas en: {os.path.join(OUTPUT_DIR, 'validation_metrics.json')}")