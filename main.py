#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualização comparativa dos quatro modelos junto com dados originais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_data
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def calculate_mse(df, test_size=60):
    """
    Calcula o MSE para cada modelo usando os últimos 'test_size' dias como conjunto de teste
    
    Args:
        df: DataFrame com os dados históricos
        test_size: Número de dias a usar como teste
    
    Returns:
        dict: Dicionário com o MSE de cada modelo
    """
    # Verificar se temos dados suficientes
    if len(df) <= test_size:
        print("Aviso: Poucos dados para cálculo do MSE. Usando todos os dados disponíveis.")
        test_size = len(df) // 2  # Usar metade dos dados
    
    results = {}
    
    # Dados para avaliação
    evaluation_data = df.copy()
    test_data = evaluation_data.iloc[-test_size:]
    
    # 1. MSE para modelo Naive
    naive_predictions = []
    for i in range(1, len(test_data)):
        # Previsão naive: usar o valor do dia anterior
        naive_predictions.append(test_data['Close'].iloc[i-1])
    
    naive_mse = mean_squared_error(test_data['Close'].iloc[1:], naive_predictions)
    results['Naive'] = naive_mse
    
    # 2. MSE para modelo MA(10)
    ma_predictions = []
    for i in range(10, len(test_data)):
        # Previsão MA: usar média dos 10 dias anteriores
        window = test_data['Close'].iloc[i-10:i]
        ma_predictions.append(window.mean())
    
    ma_mse = mean_squared_error(test_data['Close'].iloc[10:], ma_predictions)
    results['MA'] = ma_mse
    
    # 3. MSE para modelo de Regressão Linear
    # Para a regressão, podemos calcular usando a previsão de cada ponto 
    # baseada na tendência linear dos dados anteriores
    reg_predictions = []
    for i in range(10, len(test_data)):
        # Usar últimos 10 dias para a regressão
        X_train = np.arange(10).reshape(-1, 1)
        y_train = test_data['Close'].iloc[i-10:i].values
        
        # Treinar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Prever próximo ponto
        next_point = model.predict(np.array([[10]]))[0]
        reg_predictions.append(next_point)
    
    reg_mse = mean_squared_error(test_data['Close'].iloc[10:], reg_predictions)
    results['Regression'] = reg_mse
    
    return results

def plot_all_models(df, days_to_show=90):
    """
    Plota os dados originais e as previsões dos quatro modelos a partir dos arquivos CSV
    """
    # Carregar as previsões dos arquivos CSV
    naive_predictions = pd.read_csv('models_output/predictions_naive.csv', index_col=0, parse_dates=True)
    ma_predictions = pd.read_csv('models_output/predictions_ma.csv', index_col=0, parse_dates=True)
    reg_predictions = pd.read_csv('models_output/predictions_regression.csv', index_col=0, parse_dates=True)
    
    # Verificar se existe o arquivo de previsões do modelo SES
    ses_file = 'models_output/predictions_ses.csv'
    if os.path.exists(ses_file):
        ses_predictions = pd.read_csv(ses_file, index_col=0, parse_dates=True)
        has_ses = True
    else:
        has_ses = False
        print("Aviso: Arquivo de previsões do modelo SES não encontrado.")
    
    # Calcular MSE dos modelos usando os dados dos CSVs
    print("\nCalculando MSE dos modelos...")
    mse_results = {}
    
    # MSE Naive
    common_dates = df.index.intersection(naive_predictions.index)
    mse_results['Naive'] = mean_squared_error(df.loc[common_dates, 'Close'], naive_predictions.loc[common_dates, 'Prediction'])
    
    # MSE MA
    common_dates = df.index.intersection(ma_predictions.index)
    mse_results['MA'] = mean_squared_error(df.loc[common_dates, 'Close'], ma_predictions.loc[common_dates, 'Prediction'])
    
    # MSE Regressão
    common_dates = df.index.intersection(reg_predictions.index)
    mse_results['Regression'] = mean_squared_error(df.loc[common_dates, 'Close'], reg_predictions.loc[common_dates, 'Prediction'])
    
    # MSE SES (se disponível)
    if has_ses:
        common_dates = df.index.intersection(ses_predictions.index)
        mse_results['SES'] = mean_squared_error(df.loc[common_dates, 'Close'], ses_predictions.loc[common_dates, 'Prediction'])
    
    # Criar DataFrame com as previsões para o próximo dia
    next_date = naive_predictions.index[-1] + pd.Timedelta(days=1)
    future_predictions = {
        'Date': [next_date],
        'Naive': [naive_predictions['Prediction'].iloc[-1]],
        'MA': [ma_predictions['Prediction'].iloc[-1]],
        'Regression': [reg_predictions['Prediction'].iloc[-1]]
    }
    
    if has_ses:
        future_predictions['SES'] = [ses_predictions['Prediction'].iloc[-1]]
    
    future_predictions_df = pd.DataFrame(future_predictions).set_index('Date')
    
    # Preparar dados para plotagem
    recent_data = df.iloc[-days_to_show:].copy()
    
    # Criar figura
    plt.figure(figsize=(12, 8))
    
    # Plotar dados históricos
    plt.plot(recent_data.index, recent_data['Close'], 
             label='Dados Históricos', color='blue', linewidth=2)
    
    # Filtrar apenas dados recentes para o gráfico
    start_date = recent_data.index[0]
    
    # Plotar as linhas de previsão para o histórico
    naive_recent = naive_predictions[naive_predictions.index >= start_date]
    plt.plot(naive_recent.index, naive_recent['Prediction'], 
             label=f'Previsão Naive (MSE: {mse_results["Naive"]:.4f})', 
             color='red', linestyle='--', alpha=0.7)
    
    ma_recent = ma_predictions[ma_predictions.index >= start_date]
    plt.plot(ma_recent.index, ma_recent['Prediction'], 
             label=f'Previsão MA(10) (MSE: {mse_results["MA"]:.4f})', 
             color='green', linestyle='--', alpha=0.7)
    
    reg_recent = reg_predictions[reg_predictions.index >= start_date]
    plt.plot(reg_recent.index, reg_recent['Prediction'], 
             label=f'Previsão Regressão (MSE: {mse_results["Regression"]:.4f})', 
             color='magenta', linestyle='--', alpha=0.7)
    
    # Plotar SES se disponível
    if has_ses:
        ses_recent = ses_predictions[ses_predictions.index >= start_date]
        plt.plot(ses_recent.index, ses_recent['Prediction'], 
                label=f'Previsão SES (MSE: {mse_results["SES"]:.4f})', 
                color='orange', linestyle='--', alpha=0.7)
    
    # Plotar previsões para o próximo dia
    plt.plot(future_predictions_df.index, future_predictions_df['Naive'], 
             'ro', markersize=8)
    plt.plot(future_predictions_df.index, future_predictions_df['MA'], 
             'go', markersize=8)
    plt.plot(future_predictions_df.index, future_predictions_df['Regression'], 
             'mo', markersize=8)
    
    if has_ses:
        plt.plot(future_predictions_df.index, future_predictions_df['SES'], 
                'o', color='orange', markersize=8)
    
    # Adicionar linhas verticais tracejadas para destacar a previsão
    plt.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.7)
    
    # Usar formatação de data para o eixo X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Adicionar grade, legenda e títulos
    plt.grid(True, alpha=0.3)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço de Fechamento', fontsize=12)
    plt.title('Comparação de Modelos de Previsão - CPTS11', fontsize=14)
    plt.legend(loc='best')
    
    # Rotacionar os rótulos do eixo X
    plt.xticks(rotation=45)
    
    # Adicionar linha horizontal para o último preço conhecido
    plt.axhline(y=df['Close'].iloc[-1], color='blue', linestyle='--', alpha=0.5)
    
    # Adicionar anotações com os valores das previsões
    last_price = df['Close'].iloc[-1]
    plt.annotate(f'Último: {last_price:.2f}', 
                xy=(df.index[-1], last_price),
                xytext=(10, -20),
                textcoords='offset points',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    
    # Modelos e cores para anotações
    models_colors = [
        ('Naive', 'red'), 
        ('MA', 'green'), 
        ('Regression', 'magenta')
    ]
    
    if has_ses:
        models_colors.append(('SES', 'orange'))
    
    for model, color in models_colors:
        pred_value = future_predictions_df[model].iloc[0]
        plt.annotate(f'{model}: {pred_value:.2f}', 
                    xy=(future_predictions_df.index[0], pred_value),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8),
                    color=color)
    
    plt.tight_layout()
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/comparacao_modelos.png', dpi=300, bbox_inches='tight')
    
    # Mostrar o gráfico
    plt.show()
    
    # Mostrar tabela de resumo das previsões
    print("\n=== RESUMO DAS PREVISÕES ===")
    print(f"Data da previsão: {next_date.date()}")
    print(f"Último preço conhecido: {last_price:.4f}")
    print(f"Previsão Naive: {future_predictions_df['Naive'].iloc[0]:.4f} (var: {future_predictions_df['Naive'].iloc[0]-last_price:.4f})")
    print(f"Previsão MA(10): {future_predictions_df['MA'].iloc[0]:.4f} (var: {future_predictions_df['MA'].iloc[0]-last_price:.4f})")
    print(f"Previsão Regressão: {future_predictions_df['Regression'].iloc[0]:.4f} (var: {future_predictions_df['Regression'].iloc[0]-last_price:.4f})")
    
    if has_ses:
        print(f"Previsão SES: {future_predictions_df['SES'].iloc[0]:.4f} (var: {future_predictions_df['SES'].iloc[0]-last_price:.4f})")
    
    print("\n=== MÉTRICAS DE ERRO (MSE) ===")
    for model, mse in mse_results.items():
        print(f"{model}: {mse:.6f}")
    
    # Identificar melhor modelo com base no MSE
    best_model = min(mse_results, key=mse_results.get)
    print(f"\nMelhor modelo (menor MSE): {best_model}")

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    print(f"Dados carregados: {data.shape[0]} observações")
    
    # Plotar dados e previsões dos modelos
    plot_all_models(data)