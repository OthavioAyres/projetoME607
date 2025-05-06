#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo Prophet para previsão de preços do CPTS11
"""

import pandas as pd
import numpy as np
import sys, os
import warnings
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data

def predict_prophet(df, periods=1):
    """
    Utiliza o modelo Prophet para prever preços futuros
    
    Args:
        df: DataFrame pandas com os dados históricos
        periods: Número de períodos (dias) para prever no futuro
    
    Returns:
        dict: Dicionário com data e valor da previsão
    """
    # Suprimir avisos do Prophet
    warnings.filterwarnings('ignore')
    
    # Preparar dados no formato exigido pelo Prophet (ds, y)
    prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Inicializar e treinar o modelo
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # Flexibilidade para mudanças na tendência
        seasonality_prior_scale=10     # Força do componente sazonal
    )
    
    model.fit(prophet_df)
    
    # Criar dataframe para datas futuras
    future = model.make_future_dataframe(periods=periods, freq='B')  # Dias úteis (business days)
    
    # Fazer previsão
    forecast = model.predict(future)
    
    # Extrair a previsão para o próximo dia
    next_date = forecast['ds'].iloc[-1]
    next_prediction = forecast['yhat'].iloc[-1]
    
    # Mostrar componentes do modelo
    fig = model.plot_components(forecast)
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/prophet_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    print(f"Modelo Prophet: previsão para {next_date.date()} = {next_prediction:.4f}")
    
    return {
        'date': next_date,
        'forecast': next_prediction
    }

def generate_historical_predictions(df):
    """
    Gera previsões para todos os dias históricos usando o modelo Prophet
    
    Args:
        df: DataFrame pandas com os dados históricos
    
    Returns:
        DataFrame: DataFrame com as previsões históricas
    """
    # Suprimir avisos do Prophet
    warnings.filterwarnings('ignore')
    
    # Preparar resultado final
    result_df = pd.DataFrame(index=df.index)
    result_df['Close'] = df['Close']
    result_df['Prediction'] = np.nan
    
    # Usar uma janela rolante para fazer previsões históricas um dia à frente
    # Para cada dia t, treinar com dados até t-1 e prever para t
    min_train_size = 60  # Mínimo de dados necessários para treinamento
    
    if len(df) <= min_train_size:
        print("Poucos dados para previsões históricas. Necessário pelo menos 60 observações.")
        return result_df
    
    # Loop nas datas a partir de min_train_size
    for i in range(min_train_size, len(df)):
        train_data = df.iloc[:i].copy().reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Treinar modelo
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        model.fit(train_data)
        
        # Criar dataframe para datas futuras (apenas 1 dia)
        future = model.make_future_dataframe(periods=1, freq='B')
        
        # Fazer previsão
        forecast = model.predict(future)
        
        # Obter a previsão para o dia atual
        next_date = df.index[i]
        prediction = forecast['yhat'].iloc[-1]
        
        # Armazenar previsão
        result_df.loc[next_date, 'Prediction'] = prediction
        
        # Mostrar progresso
        if i % 20 == 0:
            print(f"Processando... {i}/{len(df)} ({i/len(df)*100:.1f}%)")
    
    # Remover linhas sem previsão
    result_df = result_df.dropna()
    
    # Calcular métricas de erro
    mse = mean_squared_error(result_df['Close'], result_df['Prediction'])
    mae = mean_absolute_error(result_df['Close'], result_df['Prediction'])
    rmse = np.sqrt(mse)
    
    print(f"\nMétricas de erro do modelo Prophet:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Salvar em CSV
    output_dir = 'models_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    result_df.to_csv(f'{output_dir}/predictions_prophet.csv')
    print(f"Previsões Prophet salvas em '{output_dir}/predictions_prophet.csv'")
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    return result_df

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    print(f"Dados carregados: {data.shape[0]} observações")
    
    # Fazer previsão para o próximo dia
    prediction = predict_prophet(data)
    print(f"\nPrevisão Prophet: {prediction['forecast']:.4f} para {prediction['date'].date()}")
    
    # Gerar e salvar previsões históricas
    print("\nGerando previsões históricas...")
    historical_predictions = generate_historical_predictions(data) 