#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo de Suavização Exponencial Simples para previsão de preços do CPTS11
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data

def predict_exponential_smoothing(df, alpha=0.3):
    """
    Previsão com Suavização Exponencial Simples:
    Usa uma média ponderada dos valores passados, com pesos decaindo exponencialmente
    para valores mais antigos.
    
    Args:
        df: DataFrame com os dados históricos
        alpha: Parâmetro de suavização (0 < alpha < 1)
            - Valores menores de alpha dão mais peso a observações passadas
            - Valores maiores de alpha dão mais peso a observações recentes
    
    Returns:
        dict: Dicionário com a data e valor da previsão
    """
    # Verificar que alpha está no intervalo válido
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha deve estar entre 0 e 1 (exclusivo)")
    
    # Calcular a previsão com suavização exponencial
    prices = df['Close'].values
    
    # A última previsão será usada para o próximo dia
    forecast = prices[-1]  # Inicializa com o último valor conhecido
    
    # Aplicar suavização exponencial para calcular a previsão
    for t in range(len(prices)-2, -1, -1):
        forecast = alpha * prices[t] + (1 - alpha) * forecast
    
    # Inverter para obter a previsão para o próximo dia
    # Na suavização exponencial simples, a previsão é:
    # F(t+1) = alpha * Y(t) + (1 - alpha) * F(t)
    next_forecast = alpha * prices[-1] + (1 - alpha) * forecast
    
    # Determinar a próxima data (próximo dia útil)
    last_date = df.index[-1]
    next_date = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1, freq='B')[0]
    
    print(f"Modelo SES: previsão para {next_date.date()} = {next_forecast:.4f} (alpha={alpha})")
    
    return {
        'date': next_date,
        'forecast': next_forecast,
        'alpha': alpha
    }

def generate_historical_predictions(df, alpha=0.3):
    """
    Gera previsões para todos os dias históricos usando suavização exponencial simples
    e salva em CSV
    
    Args:
        df: DataFrame com os dados históricos
        alpha: Parâmetro de suavização (0 < alpha < 1)
    
    Returns:
        DataFrame: DataFrame com as previsões
    """
    # Criar cópia do DataFrame
    predictions = df.copy()
    
    # Implementação da suavização exponencial simples
    prices = df['Close'].values
    forecasts = np.zeros(len(prices))
    
    # Inicializar com o primeiro valor
    forecasts[0] = prices[0]
    
    # Calcular as previsões
    for t in range(1, len(prices)):
        forecasts[t] = alpha * prices[t-1] + (1 - alpha) * forecasts[t-1]
    
    # Adicionar previsões ao DataFrame
    predictions['Prediction'] = forecasts
    
    # Criar pasta de saída se não existir
    output_dir = 'models_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar em CSV
    predictions.to_csv(f'{output_dir}/predictions_ses.csv')
    print(f"Previsões de Suavização Exponencial Simples salvas em '{output_dir}/predictions_ses.csv'")
    
    return predictions

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    
    # Fazer previsão
    prediction = predict_exponential_smoothing(data)
    print(f"\nPrevisão SES: {prediction['forecast']:.4f} para {prediction['date'].date()} (alpha={prediction['alpha']})")
    
    # Gerar e salvar previsões históricas
    historical_predictions = generate_historical_predictions(data) 