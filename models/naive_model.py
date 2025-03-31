#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo Ingênuo (Naive) simplificado para previsão de preços do CPTS11
"""

import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data

def predict_naive(df):
    """
    Previsão naive: usa o último valor observado como previsão para o próximo dia
    """
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    next_date = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1, freq='B')[0]
    
    print(f"Modelo Naive: previsão para {next_date.date()} = {last_price:.4f}")
    
    return {
        'date': next_date,
        'forecast': last_price
    }

def generate_historical_predictions(df):
    """Gera previsões para todos os dias históricos e salva em CSV"""
    # Criar cópia do DataFrame
    predictions = df.copy()
    
    # Previsão naive: usar o valor do dia anterior
    predictions['Prediction'] = predictions['Close'].shift(1)
    
    # Remover linhas sem previsão (primeiro dia)
    predictions = predictions.dropna()
    
    # Salvar em CSV
    predictions.to_csv('models_output/predictions_naive.csv')
    print(f"Previsões Naive salvas em 'models_output/predictions_naive.csv'")
    
    return predictions

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    
    # Fazer previsão
    prediction = predict_naive(data)
    print(f"\nPrevisão Naive: {prediction['forecast']:.4f} para {prediction['date'].date()}")
    
    # Gerar e salvar previsões históricas
    historical_predictions = generate_historical_predictions(data)