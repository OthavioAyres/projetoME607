#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo de Médias Móveis simplificado para previsão de preços do CPTS11
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data

def predict_moving_average(df, window=10):
    """
    Previsão com média móvel: usa a média dos últimos 'window' dias como previsão
    """
    # Usar apenas os últimos 'window' dias
    recent_data = df.iloc[-window:]
    
    # Calcular a média
    avg_price = recent_data['Close'].mean()
    
    # Determinar a próxima data
    last_date = df.index[-1]
    next_date = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1, freq='B')[0]
    
    print(f"Modelo MA({window}): previsão para {next_date.date()} = {avg_price:.4f}")
    
    return {
        'date': next_date,
        'forecast': avg_price
    }

def generate_historical_predictions(df):
    """Gera previsões para todos os dias históricos e salva em CSV"""
    # Criar cópia do DataFrame
    predictions = df.copy()
    
    # Previsão MA: usar média dos 10 dias anteriores
    predictions['Prediction'] = predictions['Close'].rolling(window=10).mean().shift(1)
    
    # Remover linhas sem previsão (primeiros 10 dias)
    predictions = predictions.dropna()
    
    # Salvar em CSV
    predictions.to_csv('models_output/predictions_ma.csv')
    print(f"Previsões MA salvas em 'models_output/predictions_ma.csv'")
    
    return predictions

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    
    # Fazer previsão com janela de 10 dias
    prediction = predict_moving_average(data, window=10)
    print(f"\nPrevisão MA(10): {prediction['forecast']:.4f} para {prediction['date'].date()}")
    
    # Gerar e salvar previsões históricas
    historical_predictions = generate_historical_predictions(data)