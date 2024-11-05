#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from ai_utils_text import ModelConfig, ModelClassifier, getDevice

CONFIG_FILE = "model_config.json"


def clearScreen():
    if os.name == 'nt':  # Si estás en Windows
        os.system('cls')
    else:  # Si estás en Linux o macOS
        os.system('clear')


clearScreen()


def predict_text(text: str, model: nn.Module, tokenizer, device: torch.device, config: ModelConfig, label_encoder):
    model.eval()  # Pone el modelo en modo de evaluación
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
    confidence = confidence.item()
    return predicted_label, confidence


def main():
    # Cargar la configuración
    with open(CONFIG_FILE) as f:
        config_file = json.load(f)

    # Cargar las metadatos
    with open(config_file['paths']['metadata'], 'r') as f:
        metadata = json.load(f)
    labels = metadata["categories"]

    # Configuración del modelo
    config = ModelConfig(config_file, labels)

    # Inicializar tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Cargar el LabelEncoder y configurarlo con las etiquetas
    le = LabelEncoder()
    le.fit(metadata['label_encoder'])  # Inicializa correctamente las clases

    # Cargar el modelo entrenado
    device = getDevice()
    model = ModelClassifier(config).to(device)
    model.load_state_dict(torch.load(config_file['paths']['trained_network'], map_location=device, weights_only=True))
    model.eval()

    # Obtener entrada del usuario y predecir la etiqueta
    linea = input("What's your opinion about the airline?\n")

    # Utilizar la función predict_text para hacer la predicción
    predicted_label, confidence = predict_text(linea, model, tokenizer, device, config, le)

    # Mostrar el resultado
    print(f"\nPrediction: {predicted_label} with confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
