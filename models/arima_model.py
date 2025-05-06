#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo ARIMA para previsão de preços do CPTS11
"""

import pandas as pd
import numpy as np
import sys, os
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

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

def encontrar_melhores_parametros(serie, p_max=2, d_max=2, q_max=2):
    """
    Encontra os melhores parâmetros (p,d,q) para o modelo ARIMA
    usando busca em grade e AIC (Akaike Information Criterion)
    
    Args:
        serie: Série temporal para análise
        p_max: Valor máximo para p (AR)
        d_max: Valor máximo para d (I)
        q_max: Valor máximo para q (MA)
    
    Returns:
        tuple: Melhores parâmetros (p,d,q)
    """
    # Verificar estacionariedade para sugerir valor mínimo de d
    estacionaria = verificar_estacionariedade(serie)
    d_min = 0 if estacionaria else 1
    
    # Valores para busca
    p_range = range(0, p_max + 1)
    d_range = range(d_min, d_max + 1)
    q_range = range(0, q_max + 1)
    
    # Melhor modelo até agora
    melhor_aic = float('inf')
    melhor_ordem = None
    
    # Contador para acompanhar o progresso
    total_combinacoes = (p_max + 1) * (d_max - d_min + 1) * (q_max + 1)
    combinacoes_testadas = 0
    
    print(f"\n=== Otimizando parâmetros ARIMA ===")
    print(f"Testando {total_combinacoes} combinações de p, d, q...")
    
    # Testar todas as combinações
    for p, d, q in itertools.product(p_range, d_range, q_range):
        combinacoes_testadas += 1
        
        # Não testar modelos com p=0 e q=0 (não é ARIMA)
        if p == 0 and q == 0:
            continue
            
        # Mostrar progresso
        if combinacoes_testadas % 5 == 0 or combinacoes_testadas == total_combinacoes:
            print(f"Progresso: {combinacoes_testadas}/{total_combinacoes} ({combinacoes_testadas/total_combinacoes*100:.1f}%)")
        
        try:
            # Ajustar modelo
            modelo = ARIMA(serie, order=(p, d, q))
            resultado = modelo.fit()
            
            # Avaliar modelo
            aic = resultado.aic
            
            # Atualizar melhor modelo
            if aic < melhor_aic:
                melhor_aic = aic
                melhor_ordem = (p, d, q)
                print(f"Novo melhor modelo: ARIMA{melhor_ordem} (AIC: {melhor_aic:.4f})")
                
        except Exception as e:
            # Ignorar modelos que não convergem
            continue
    
    if melhor_ordem is None:
        print("Não foi possível encontrar um modelo adequado. Usando valores padrão.")
        melhor_ordem = (1, d_min, 1)
    
    print(f"\nMelhores parâmetros encontrados: ARIMA{melhor_ordem} (AIC: {melhor_aic:.4f})")
    return melhor_ordem

def predict_arima(df, otimizar_parametros=True, order=(2,1,2)):
    """
    Utiliza o modelo ARIMA para prever o próximo valor da série
    
    Args:
        df: DataFrame com os dados históricos
        otimizar_parametros: Se True, otimiza os parâmetros p, d, q ** Othavio: (2, 1, 2) foi o obtido com otimização
        order: Ordem do modelo ARIMA (p,d,q) se não otimizar
    
    Returns:
        dict: Dicionário com data e valor da previsão
    """
    # Suprimir avisos do statsmodels
    warnings.filterwarnings('ignore')
    
    # Verificar estacionariedade da série
    estacionaria = verificar_estacionariedade(df['Close'])
    
    # Determinar os parâmetros do modelo
    if otimizar_parametros:
        print("\nOtimizando parâmetros do modelo ARIMA...")
        order = encontrar_melhores_parametros(df['Close'])
    else:
        # Se não for estacionária, sugerir diferenciar a série
        d = order[1]
        if not estacionaria and d == 0:
            print("AVISO: Série não estacionária. Recomenda-se usar d ≥ 1.")
            d = 1
            order = (order[0], d, order[2])
    
    print(f"\nAjustando modelo ARIMA{order}...")
    
    # Treinar modelo ARIMA
    modelo = ARIMA(df['Close'], order=order)
    resultado = modelo.fit()
    
    # Resumo do modelo
    print("\n=== Resumo do Modelo ARIMA ===")
    print(resultado.summary().tables[0].as_text())
    print("\nParâmetros estimados:")
    print(resultado.summary().tables[1].as_text())
    
    # Fazer previsão para o próximo dia
    forecast = resultado.forecast(steps=1)
    forecast_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast
    next_date = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=1, freq='B')[0]
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    print(f"\nModelo ARIMA{order}: previsão para {next_date.date()} = {forecast_value:.4f}")
    
    return {
        'date': next_date,
        'forecast': forecast_value,
        'order': order
    }

def generate_historical_predictions(df, otimizar_parametros=True):
    """
    Gera previsões históricas usando o modelo ARIMA
    
    Args:
        df: DataFrame com os dados históricos
        otimizar_parametros: Se True, otimiza os parâmetros p, d, q
        
    Returns:
        DataFrame: DataFrame com as previsões históricas
    """
    # Suprimir avisos do statsmodels
    warnings.filterwarnings('ignore')
    
    # Criar resultado
    result_df = pd.DataFrame(index=df.index)
    result_df['Close'] = df['Close']
    result_df['Prediction'] = np.nan
    
    # Determinar o modelo a ser usado
    if otimizar_parametros:
        # Usar primeiros 80% dos dados para otimização
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        
        print("\nOtimizando parâmetros do modelo ARIMA usando 80% dos dados...")
        order = encontrar_melhores_parametros(train_data['Close'])
    else:
        # Usando ordem fixa (2,1,2)
        order = (2, 1, 2)
    
    print(f"\nGerando previsões históricas com ARIMA{order}...")
    
    # Tamanho mínimo da janela de treinamento
    min_train_size = max(60, 2 * sum(order))  # Pelo menos 60 observações ou 2x a soma das ordens
    
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
            # Treinar modelo
            model = ARIMA(train_data['Close'], order=order)
            result = model.fit()
            
            # Prever próximo valor
            forecast = result.forecast(steps=1)
            forecast_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast
            
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
    
    print(f"\nMétricas de erro do modelo ARIMA{order}:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Salvar em CSV
    output_dir = 'models_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    result_df.to_csv(f'{output_dir}/predictions_arima.csv')
    print(f"Previsões ARIMA salvas em '{output_dir}/predictions_arima.csv'")
    
    # Salvar informações do modelo
    with open(f'{output_dir}/arima_info.txt', 'w') as f:
        f.write(f"Parâmetros ARIMA: {order}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
    
    # Plotar predições vs valores reais para avaliação visual
    plt.figure(figsize=(12, 6))
    plt.plot(result_df.index[-60:], result_df['Close'][-60:], label='Real', color='blue')
    plt.plot(result_df.index[-60:], result_df['Prediction'][-60:], label=f'Predição ARIMA{order}', color='red', linestyle='--')
    plt.title(f'ARIMA{order} - Previsões vs Valores Reais (Últimos 60 dias)')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/arima_predictions.png', dpi=300, bbox_inches='tight')
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    return result_df

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    print(f"Dados carregados: {data.shape[0]} observações")
    
    # Fazer previsão para o próximo dia com parâmetros otimizados
    prediction = predict_arima(data, otimizar_parametros=False)
    print(f"\nPrevisão ARIMA{prediction['order']}: {prediction['forecast']:.4f} para {prediction['date'].date()}")
    
    # Gerar e salvar previsões históricas
    print("\nGerando previsões históricas...")
    historical_predictions = generate_historical_predictions(data, otimizar_parametros=True)