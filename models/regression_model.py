#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo de Regressão Linear simples para previsão do próximo dia
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data
from sklearn.linear_model import LinearRegression

def predict_regression(df):
    """
    Realiza regressão linear simples para prever o próximo dia
    """
    # Criar feature simples: índice numérico (representa tendência temporal)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    
    # Treinar modelo com todos os dados
    model = LinearRegression()
    model.fit(X, y)
    
    # Prever o próximo dia (índice = len(df))
    next_day_index = np.array([[len(df)]])
    forecast = model.predict(next_day_index)[0]
    
    # Determinar a data do próximo dia útil
    last_date = df.index[-1]
    next_date = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                              periods=1, freq='B')[0]
    
    print(f"Regressão Linear - Coeficiente angular: {model.coef_[0]:.6f}")
    print(f"Regressão Linear - Intercepto: {model.intercept_:.6f}")
    return {
        'date': next_date,
        'forecast': forecast,
        'model': model
    }

def generate_historical_predictions(df):
    """Gera previsões para todos os dias históricos usando regressão linear simples e salva em CSV"""
    # Criar cópia do DataFrame
    predictions = df.copy()
    
    # Treinar modelo com todos os dados
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Gerar previsões para cada ponto
    predictions['Prediction'] = model.predict(X)
    
    # Salvar em CSV
    predictions.to_csv('models_output/predictions_regression.csv')
    print(f"Previsões Regressão salvas em 'models_output/predictions_regression.csv'")
    
    print(f"Coeficiente angular: {model.coef_[0]:.6f}")
    print(f"Intercepto: {model.intercept_:.6f}")
    
    return predictions

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    
    # Fazer previsão com todos os dados
    prediction = predict_regression(data)
    print(f"\nPrevisão Regressão: {prediction['forecast']:.4f} para {prediction['date'].date()}")
    
    # Gerar e salvar previsões históricas
    historical_predictions = generate_historical_predictions(data)