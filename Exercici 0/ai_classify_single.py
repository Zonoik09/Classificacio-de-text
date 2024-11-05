#!/usr/bin/env python3

import os
import json
print("Loading AI libraries ...");
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from typing import List
from ai_utils_text import ModelConfig, ModelDataset, ModelClassifier, getDevice

CONFIG_FILE = "model_config.json"

def clearScreen():
    if os.name == 'nt':     # Si estàs a Windows
        os.system('cls')
    else:                   # Si estàs a Linux o macOS
        os.system('clear')

clearScreen()

def predict_text(text: str, model: nn.Module, tokenizer, device: torch.device, config: ModelConfig, label_encoder):
    model.eval()  # Posa el model en mode d'avaluació
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

    # Cargar las metadades generades durant l'entrenament
    with open(config_file['paths']['metadata'], 'r') as f:
        metadata = json.load(f)
    labels = metadata["categories"]

    # Configuración del modelo
    config = ModelConfig(config_file, labels)

    # Inicializar tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Cargar el label encoder para las categorías
    le = LabelEncoder()
    le.fit(metadata['label_encoder'])

    # Cargar el modelo entrenado
    device = getDevice()
    model = ModelClassifier(config).to(device)
    model.load_state_dict(torch.load(config_file['paths']['trained_network'], map_location=device, weights_only=True))
    model.eval()

    # Solicitar una frase al usuario
    text = input("Por favor, ingresa una frase para verificar si está en inglés: ")

    # Realizar la predicción
    predicted_label, confidence = predict_text(text, model, tokenizer, device, config, le)

    # Mostrar el resultado de la predicción
    print(f"\nFrase: \"{text}\"\nPredicción: {predicted_label} (confianza: {confidence:.2%})")

    if predicted_label != "other":
        print("This is English")
    else:
        print("I don't understand you")

if __name__ == "__main__":
    main()
