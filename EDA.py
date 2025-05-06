#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análise Exploratória de Dados - CPTS11
Este script calcula estatísticas descritivas e gera o correlograma dos dados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_data

def calcular_estatisticas_descritivas(df):
    """
    Calcula e exibe estatísticas descritivas dos dados.
    
    Args:
        df: DataFrame com os dados históricos
    
    Returns:
        df_stats: DataFrame com as estatísticas calculadas
    """
    # Verificar se existem valores faltantes
    print("\n=== VERIFICAÇÃO DE VALORES FALTANTES ===")
    valores_faltantes = df['Close'].isna().sum()
    percentual_faltantes = (valores_faltantes / len(df)) * 100
    
    print(f"Total de valores faltantes: {valores_faltantes}")
    print(f"Percentual de valores faltantes: {percentual_faltantes:.2f}%")
    
    if valores_faltantes > 0:
        print("\nPeríodos com valores faltantes:")
        datas_faltantes = df[df['Close'].isna()].index
        for data in datas_faltantes:
            print(f"  - {data.date()}")
    
    print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    
    # Calcular estatísticas básicas
    stats = df['Close'].describe()
    print(stats)
    
    # Calcular estatísticas adicionais
    additional_stats = {
        'Variância': df['Close'].var(),
        'Assimetria': df['Close'].skew(),
        'Curtose': df['Close'].kurtosis(),
        'Primeiro valor': df['Close'].iloc[0],
        'Último valor': df['Close'].iloc[-1],
        'Variação (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100,
        'Retorno médio diário (%)': df['Close'].pct_change().mean() * 100,
        'Volatilidade diária (%)': df['Close'].pct_change().std() * 100
    }
    
    print("\n=== ESTATÍSTICAS ADICIONAIS ===")
    for stat, value in additional_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # Criar um dataframe completo para salvar/retornar
    df_stats = pd.DataFrame(stats).T
    for stat, value in additional_stats.items():
        df_stats[stat] = value
    
    # Adicionar informações sobre valores faltantes
    df_stats['Valores_Faltantes'] = valores_faltantes
    df_stats['Percentual_Faltantes'] = percentual_faltantes
    
    return df_stats

def plotar_serie_temporal(df, dias_para_mostrar=None):
    """
    Plota a série temporal dos preços de fechamento.
    
    Args:
        df: DataFrame com os dados históricos
        dias_para_mostrar: Número de dias recentes para mostrar (opcional)
    """
    # Criar uma cópia para evitar SettingWithCopyWarning
    data_plot = df.copy()
    
    if dias_para_mostrar:
        data_plot = data_plot.iloc[-dias_para_mostrar:]

    plt.figure(figsize=(12, 6))
    
    # Plotar preço de fechamento
    plt.plot(data_plot.index, data_plot['Close'], label='Preço de Fechamento', color='blue')
    
    # Adicionar média móvel de 20 dias
    data_plot['MA20'] = data_plot['Close'].rolling(window=20).mean()
    plt.plot(data_plot.index, data_plot['MA20'], label='Média Móvel (20 dias)', color='red', linestyle='--')
    
    # Configurar eixos e títulos
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço de Fechamento', fontsize=12)
    plt.title('Série Temporal - CPTS11', fontsize=14)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/serie_temporal.png', dpi=300, bbox_inches='tight')
    
    # Mostrar gráfico
    plt.show()

def plotar_correlograma(df, lags=40):
    """
    Plota o correlograma (autocorrelação) e autocorrelação parcial dos dados.
    
    Args:
        df: DataFrame com os dados históricos
        lags: Número de lags para exibir no correlograma
    """
    warnings.filterwarnings('ignore')  # Ignorar avisos do statsmodels
    
    # Criar figura com dois subplots verticais
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calcular e plotar a função de autocorrelação
    plot_acf(df['Close'], lags=lags, ax=ax1, alpha=0.05)
    ax1.set_title('Função de Autocorrelação (ACF)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Calcular e plotar a função de autocorrelação parcial
    plot_pacf(df['Close'], lags=lags, ax=ax2, alpha=0.05, method='ywm')
    ax2.set_title('Função de Autocorrelação Parcial (PACF)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/correlograma_series_cpts11.png', dpi=300, bbox_inches='tight')
    
    # Mostrar gráfico
    plt.show()
    
    # Reativar avisos
    warnings.filterwarnings('default')

def plotar_distribuicao_retornos(df):
    """
    Plota a distribuição dos retornos diários.
    
    Args:
        df: DataFrame com os dados históricos
    """
    # Calcular retornos diários
    retornos = df['Close'].pct_change().dropna()
    
    plt.figure(figsize=(12, 6))
    
    # Histograma com KDE
    sns.histplot(retornos, kde=True, color='blue', stat='density')
    
    # Adicionar distribuição normal para comparação
    x = np.linspace(retornos.min(), retornos.max(), 100)
    normal_dist = np.exp(-(x - retornos.mean())**2 / (2 * retornos.std()**2)) / (retornos.std() * np.sqrt(2 * np.pi))
    plt.plot(x, normal_dist, 'r--', label='Distribuição Normal')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Retorno Diário', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    plt.title('Distribuição dos Retornos Diários - CPTS11', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/distribuicao_retornos.png', dpi=300, bbox_inches='tight')
    
    # Mostrar gráfico
    plt.show()

def plotar_decomposicao_classica(df, periodo=20):
    """
    Realiza a decomposição clássica da série temporal em tendência, 
    sazonalidade e resíduos.
    
    Args:
        df: DataFrame com os dados históricos
        periodo: Período sazonal para decomposição (padrão: 20 dias)
    """
    warnings.filterwarnings('ignore')  # Ignorar avisos do statsmodels
    
    # Verificar se há dados suficientes
    if len(df) < 2 * periodo:
        print(f"AVISO: Série muito curta para decomposição com período {periodo}.")
        print(f"Usando período = {len(df) // 4} para possibilitar a decomposição.")
        periodo = len(df) // 4
    
    # Realizar a decomposição
    print(f"\n=== DECOMPOSIÇÃO CLÁSSICA (período = {periodo}) ===")
    decomposicao = seasonal_decompose(df['Close'], model='additive', period=periodo)
    
    # Acessar os componentes da decomposição
    tendencia = decomposicao.trend
    sazonalidade = decomposicao.seasonal
    residuos = decomposicao.resid
    
    # Criar figura para visualização dos componentes
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Dados originais
    axes[0].plot(df.index, df['Close'], label='Original', color='blue')
    axes[0].set_title('Série Original', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Tendência
    axes[1].plot(df.index, tendencia, label='Tendência', color='red')
    axes[1].set_title('Componente de Tendência', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Sazonalidade
    axes[2].plot(df.index, sazonalidade, label='Sazonalidade', color='green')
    axes[2].set_title('Componente Sazonal', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Resíduos
    axes[3].plot(df.index, residuos, label='Resíduos', color='purple')
    axes[3].set_title('Componente Residual', fontsize=14)
    axes[3].grid(True, alpha=0.3)
    
    # Configurar eixos e layout
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/decomposicao_classica.png', dpi=300, bbox_inches='tight')
    
    # Mostrar gráfico
    plt.show()
    
    # Calcular estatísticas dos componentes
    print("\n=== ESTATÍSTICAS DOS COMPONENTES ===")
    print(f"Média da tendência: {tendencia.mean():.4f}")
    print(f"Média da sazonalidade: {sazonalidade.mean():.4f}")
    print(f"Média dos resíduos: {residuos.mean():.4f}")
    print(f"Desvio padrão da tendência: {tendencia.std():.4f}")
    print(f"Desvio padrão da sazonalidade: {sazonalidade.std():.4f}")
    print(f"Desvio padrão dos resíduos: {residuos.std():.4f}")
    
    # Reativar avisos
    warnings.filterwarnings('default')
    
    # Retornar componentes para possíveis análises adicionais
    return {
        'tendencia': tendencia,
        'sazonalidade': sazonalidade,
        'residuos': residuos
    }

def executar_analise_exploratoria():
    """
    Executa a análise exploratória completa dos dados
    """
    # Carregar dados
    df = load_data()
    print(f"Dados carregados: {df.shape[0]} observações")
    print(f"Período: {df.index[0].date()} a {df.index[-1].date()}")
    
    # Calcular estatísticas descritivas
    estatisticas = calcular_estatisticas_descritivas(df)
    
    # Plotar série temporal
    plotar_serie_temporal(df)
    
    # Plotar distribuição dos retornos
    plotar_distribuicao_retornos(df)
    
    # Plotar o correlograma
    plotar_correlograma(df)
    
    # Realizar decomposição clássica da série
    plotar_decomposicao_classica(df)
    
    # Salvar estatísticas em CSV
    output_dir = 'resultados'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    estatisticas.to_csv(f'{output_dir}/estatisticas_descritivas.csv')
    
    print("\nAnálise exploratória concluída. Os gráficos foram salvos na pasta 'graficos'.")
    print("As estatísticas descritivas foram salvas em 'resultados/estatisticas_descritivas.csv'.")

if __name__ == "__main__":
    executar_analise_exploratoria() 