#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo AR(2) para previsão de preços do CPTS11

Parâmetros estimados:
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0019      0.001      2.093      0.036       0.000       0.004
Close.L1      -0.0814      0.020     -4.127      0.000      -0.120      -0.043
Close.L2       0.0704      0.020      3.566      0.000       0.032       0.109
==============================================================================
"""

import pandas as pd
import numpy as np
import sys, os
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data

def verificar_estacionariedade(serie):
    """
    Verifica se a série é estacionária usando o teste ADF
    
    Args:
        serie: Série temporal a ser testada
        
    Returns:
        bool: True se a série for estacionária, False caso contrário
    """
    resultado = adfuller(serie.dropna())
    
    print('\n=== Teste de Estacionariedade (ADF) ===')
    print(f'Estatística de teste: {resultado[0]:.4f}')
    print(f'Valor-p: {resultado[1]:.4f}')
    
    for chave, valor in resultado[4].items():
        print(f'Valor crítico ({chave}): {valor:.4f}')
    
    estacionaria = resultado[1] < 0.05
    print(f'Série {"é" if estacionaria else "não é"} estacionária (95% confiança)')
    
    return estacionaria

def predict_arima210(df):
    """
    Utiliza o modelo ARIMA(2,1,0) para prever o próximo valor da série
    
    Args:
        df: DataFrame com os dados históricos
    
    Returns:
        dict: Dicionário com data e valor da previsão
    """
    # Suprimir avisos do statsmodels
    warnings.filterwarnings('ignore')
    
    # Sempre aplicar diferenciação de primeira ordem
    print("\nAplicando diferenciação de primeira ordem (ARIMA(2,1,0))...")
    serie = df['Close'].copy()
    serie_diff = serie.diff().dropna()
    
    # Verificar estacionariedade da série diferenciada
    verificar_estacionariedade(serie_diff)
    
    print("\nAjustando modelo ARIMA(2,1,0)...")
    
    # Treinar modelo AR(2) na série diferenciada
    modelo = AutoReg(serie_diff, lags=2, trend='c')
    resultado = modelo.fit()
    
    # Resumo do modelo
    print("\n=== Resumo do Modelo ARIMA(2,1,0) ===")
    print(resultado.summary().tables[0].as_text())
    print("\nParâmetros estimados:")
    print(resultado.summary().tables[1].as_text())
    
    # Fazer previsão para o próximo dia (diferença)
    forecast_diff = resultado.forecast(steps=1)
    forecast_diff_value = forecast_diff.iloc[0] if hasattr(forecast_diff, 'iloc') else forecast_diff[0] if isinstance(forecast_diff, np.ndarray) else forecast_diff
    
    # Converter previsão de volta para o valor original
    forecast_value = df['Close'].iloc[-1] + forecast_diff_value
    
    next_date = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=1, freq='B')[0]
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    print(f"\nModelo ARIMA(2,1,0): previsão para {next_date.date()} = {forecast_value:.4f}")
    
    return {
        'date': next_date,
        'forecast': forecast_value
    }

def generate_historical_predictions_arima210(df):
    """
    Gera previsões históricas usando o modelo ARIMA(2,1,0)
    
    Args:
        df: DataFrame com os dados históricos
    
    Returns:
        DataFrame: DataFrame com as previsões históricas
    """
    # Suprimir avisos do statsmodels
    warnings.filterwarnings('ignore')
    
    print(f"\nGerando previsões históricas com ARIMA(2,1,0)...")
    
    # Criar resultado
    result_df = pd.DataFrame(index=df.index)
    result_df['Close'] = df['Close']
    result_df['Prediction'] = np.nan
    
    # Tamanho mínimo da janela de treinamento (pelo menos 30 observações para AR(2))
    min_train_size = 30
    
    if len(df) <= min_train_size:
        print(f"Poucos dados para previsões históricas. Necessário pelo menos {min_train_size} observações.")
        return result_df
    
    # Loop pelas datas para fazer previsões históricas
    for i in range(min_train_size, len(df)):
        if i % 20 == 0:
            print(f"Processando... {i}/{len(df)} ({i/len(df)*100:.1f}%)")
        
        # Usar dados até i-1 para prever i
        train_data = df.iloc[:i].copy()
        try:
            serie = train_data['Close'].copy()
            serie_diff = serie.diff().dropna()
            # Treinar modelo AR(2) na série diferenciada
            model = AutoReg(serie_diff, lags=2, trend='c')
            result = model.fit()
            # Prever próximo valor da diferença
            forecast_diff = result.forecast(steps=1)
            forecast_diff = forecast_diff.iloc[0] if hasattr(forecast_diff, 'iloc') else forecast_diff[0] if isinstance(forecast_diff, np.ndarray) else forecast_diff
            # Converter de volta para o valor original
            forecast_value = serie.iloc[-1] + forecast_diff
            # Armazenar previsão
            result_df.loc[df.index[i], 'Prediction'] = forecast_value
        except Exception as e:
            print(f"Erro ao processar data {df.index[i].date()}: {e}")
    
    # Remover linhas sem previsão
    result_df = result_df.dropna()
    
    # Calcular métricas de erro
    mse = mean_squared_error(result_df['Close'], result_df['Prediction'])
    mae = mean_absolute_error(result_df['Close'], result_df['Prediction'])
    rmse = np.sqrt(mse)
    
    print(f"\nMétricas de erro do modelo ARIMA(2,1,0):")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Salvar em CSV
    output_dir = 'models_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_df.to_csv(f'{output_dir}/predictions_arima210.csv')
    print(f"Previsões ARIMA(2,1,0) salvas em '{output_dir}/predictions_arima210.csv'")
    
    # Salvar informações do modelo
    with open(f'{output_dir}/arima210_info.txt', 'w') as f:
        f.write(f"Modelo: ARIMA(2,1,0)\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
    
    # Plotar predições vs valores reais para avaliação visual
    plt.figure(figsize=(12, 6))
    plt.plot(result_df.index[-60:], result_df['Close'][-60:], label='Real', color='blue')
    plt.plot(result_df.index[-60:], result_df['Prediction'][-60:], label='Predição ARIMA(2,1,0)', color='green', linestyle='--')
    plt.title('ARIMA(2,1,0) - Previsões vs Valores Reais (Últimos 60 dias)')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/arima210_predictions.png', dpi=300, bbox_inches='tight')
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    return result_df

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    print(f"Dados carregados: {data.shape[0]} observações")
    
    # Fazer previsão para o próximo dia
    prediction = predict_arima210(data)
    print(f"\nPrevisão ARIMA(2,1,0): {prediction['forecast']:.4f} para {prediction['date'].date()}")
    
    # Gerar e salvar previsões históricas
    print("\nGerando previsões históricas...")
    historical_predictions = generate_historical_predictions_arima210(data) 