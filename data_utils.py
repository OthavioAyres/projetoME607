#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para carregamento e preparação de dados do CPTS11
Este módulo contém funções comuns usadas por todos os modelos
"""

import pandas as pd


def load_data():
    """
    Carrega e prepara os dados do CPTS11
    
    Esta função carrega o arquivo CSV, seleciona as duas primeiras colunas,
    mantém apenas o último ano de dados, renomeia as colunas e converte
    a coluna de data para datetime, definindo-a como índice.
    
    Returns:
    -------
    DataFrame
        DataFrame com os dados preparados do CPTS11
    """
    # Carregar os dados
    df = pd.read_csv('CPTS11_new.csv') #, skiprows=2)
    
    # Começa 1/10/2023 e termina 01/04/2025
    # Manter apenas as duas primeiras colunas e apenas o último ano
    df = df.iloc[:, :2]
    
    # Renomear a segunda coluna para 'Close'
    df.columns = ['Date', 'Close']
    
    # Converter "Date" para datetime e definir como índice
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    print("df.shape", df.shape)
    
    # Converter a coluna 'Close' para numérico
    df['Close'] = pd.to_numeric(df['Close'].str.replace(',', '.'), errors='coerce')
    # Remover linhas com valores faltantes
    df = df.dropna()
    
    return df


def evaluate_model(y_true, y_pred, nome_modelo="Modelo"):
    """
    Avalia o modelo calculando métricas de erro para diferentes horizontes de previsão
    
    Parameters:
    ----------
    y_true : Series
        Valores reais
    y_pred : Series
        Valores previstos pelo modelo
    nome_modelo : str
        Nome do modelo para exibição
        
    Returns:
    -------
    dict
        Dicionário com métricas de avaliação para horizontes de 1, 5 e 10 dias
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    import pandas as pd
    
    # Garantir que os dados estão alinhados
    dados_combinados = pd.concat([y_true, y_pred], axis=1, join='inner')
    dados_combinados.columns = ['Real', 'Previsto']
    
    # Criar deslocamentos para avaliação de horizontes de previsão
    horizontes = [1]
    resultados = {}
    
    print(f"\nAvaliação do Modelo {nome_modelo}:")
    
    for h in horizontes:
        # Deslocar valores reais para simular previsão h passos à frente
        dados_h = dados_combinados.copy()
        dados_h['Real_Futuro'] = dados_h['Real'].shift(-h)
        dados_h = dados_h.dropna()
        
        if len(dados_h) == 0:
            print(f"  Horizonte de {h} dias: Dados insuficientes")
            continue
            
        # Calcular métricas
        mae = mean_absolute_error(dados_h['Real_Futuro'], dados_h['Previsto'])
        rmse = np.sqrt(mean_squared_error(dados_h['Real_Futuro'], dados_h['Previsto']))
        r2 = r2_score(dados_h['Real_Futuro'], dados_h['Previsto'])
        mape = np.mean(np.abs((dados_h['Real_Futuro'] - dados_h['Previsto']) / dados_h['Real_Futuro'])) * 100
        
        print(f"  Horizonte de {h} dia(s):")
        print(f"    MAE: {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    R²: {r2:.6f}")
        print(f"    MAPE: {mape:.2f}%")
        
        resultados[h] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    return resultados


def calcular_retorno_log(df):
    """
    Calcula os retornos logarítmicos a partir da coluna de fechamento ('Close').
    Trata valores faltantes e datas não sequenciais.

    Parâmetros:
    ----------
    df : DataFrame
        DataFrame com a coluna 'Close' (preços de fechamento)
    
    Retorna:
    -------
    Series
        Série de retornos logarítmicos
    """
    import numpy as np
    # Garante que a coluna 'Close' está ordenada por data
    df = df.sort_index()
    # Calcula o retorno logarítmico
    retorno_log = np.log(df['Close'] / df['Close'].shift(1))
    # Remove valores faltantes
    retorno_log = retorno_log.dropna()
    retorno_log.name = 'Retorno_Log'
    return retorno_log


if __name__ == "__main__":
    # Teste rápido para verificar a função
    data = load_data()
    print("Dados carregados com sucesso!")
    print(f"Formato dos dados: {data.shape}")
    print("Primeiras linhas dos dados:")
    print(data.head())
    print("\nÚltimas linhas dos dados:")
    print(data.tail()) 