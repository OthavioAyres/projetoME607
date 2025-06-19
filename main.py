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

def plot_all_models(df, days_to_show=300):
    """
    Plota os dados originais e as previsões dos modelos a partir dos arquivos CSV
    """
    # Carregar as previsões dos arquivos CSV
    naive_predictions = pd.read_csv('models_output/predictions_naive.csv', index_col=0, parse_dates=True)
    ma_predictions = pd.read_csv('models_output/predictions_ma.csv', index_col=0, parse_dates=True)
    reg_predictions = pd.read_csv('models_output/predictions_regression.csv', index_col=0, parse_dates=True)
    
    # Verificar se existe o arquivo de previsões do modelo SES
    ses_file = 'models_output/predictions_ses.csv'
    has_ses = os.path.exists(ses_file)
    if has_ses:
        ses_predictions = pd.read_csv(ses_file, index_col=0, parse_dates=True)
    else:
        print("Aviso: Arquivo de previsões do modelo SES não encontrado.")
    
    # Verificar se existe o arquivo de previsões do modelo Prophet
    prophet_file = 'models_output/predictions_prophet.csv'
    has_prophet = os.path.exists(prophet_file)
    if has_prophet:
        prophet_predictions = pd.read_csv(prophet_file, index_col=0, parse_dates=True)
    else:
        print("Aviso: Arquivo de previsões do modelo Prophet não encontrado.")
    
    # Verificar se existe o arquivo de previsões do modelo ARIMA
    arima_file = 'models_output/predictions_arima.csv'
    has_arima = os.path.exists(arima_file)
    if has_arima:
        arima_predictions = pd.read_csv(arima_file, index_col=0, parse_dates=True)
    else:
        print("Aviso: Arquivo de previsões do modelo ARIMA não encontrado.")
    
    # Verificar se existe o arquivo de previsões do modelo AR(2)
    ar2_file = 'models_output/predictions_ar2.csv'
    has_ar2 = os.path.exists(ar2_file)
    if has_ar2:
        ar2_predictions = pd.read_csv(ar2_file, index_col=0, parse_dates=True)
    else:
        print("Aviso: Arquivo de previsões do modelo AR(2) não encontrado.")
    
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
    
    # MSE Prophet (se disponível)
    if has_prophet:
        common_dates = df.index.intersection(prophet_predictions.index)
        mse_results['Prophet'] = mean_squared_error(df.loc[common_dates, 'Close'], prophet_predictions.loc[common_dates, 'Prediction'])
    
    # MSE ARIMA (se disponível)
    if has_arima:
        common_dates = df.index.intersection(arima_predictions.index)
        mse_results['ARIMA'] = mean_squared_error(df.loc[common_dates, 'Close'], arima_predictions.loc[common_dates, 'Prediction'])
    
    # MSE AR(2) (se disponível)
    if has_ar2:
        common_dates = df.index.intersection(ar2_predictions.index)
        mse_results['AR(2)'] = mean_squared_error(df.loc[common_dates, 'Close'], ar2_predictions.loc[common_dates, 'Prediction'])
    
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
    
    if has_prophet:
        future_predictions['Prophet'] = [prophet_predictions['Prediction'].iloc[-1]]
    
    if has_arima:
        future_predictions['ARIMA'] = [arima_predictions['Prediction'].iloc[-1]]
    
    if has_ar2:
        future_predictions['AR(2)'] = [ar2_predictions['Prediction'].iloc[-1]]
    
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
    
    # Plotar Prophet se disponível
    if has_prophet:
        prophet_recent = prophet_predictions[prophet_predictions.index >= start_date]
        plt.plot(prophet_recent.index, prophet_recent['Prediction'], 
                label=f'Previsão Prophet (MSE: {mse_results["Prophet"]:.4f})', 
                color='purple', linestyle='--', alpha=0.7)
    
    # Plotar ARIMA se disponível
    if has_arima:
        arima_recent = arima_predictions[arima_predictions.index >= start_date]
        plt.plot(arima_recent.index, arima_recent['Prediction'], 
                label=f'Previsão ARIMA(2,1,2) (MSE: {mse_results["ARIMA"]:.4f})', 
                color='brown', linestyle='--', alpha=0.7)
    
    # Plotar AR(2) se disponível
    if has_ar2:
        ar2_recent = ar2_predictions[ar2_predictions.index >= start_date]
        plt.plot(ar2_recent.index, ar2_recent['Prediction'], 
                label=f'Previsão AR(2) (MSE: {mse_results["AR(2)"]:.4f})', 
                color='cyan', linestyle='--', alpha=0.7)
    
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
    
    if has_prophet:
        plt.plot(future_predictions_df.index, future_predictions_df['Prophet'], 
                'o', color='purple', markersize=8)
    
    if has_arima:
        plt.plot(future_predictions_df.index, future_predictions_df['ARIMA'], 
                'o', color='brown', markersize=8)
    
    if has_ar2:
        plt.plot(future_predictions_df.index, future_predictions_df['AR(2)'], 
                'o', color='cyan', markersize=8)
    
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
    
    if has_prophet:
        models_colors.append(('Prophet', 'purple'))
    
    if has_arima:
        models_colors.append(('ARIMA', 'brown'))
    
    if has_ar2:
        models_colors.append(('AR(2)', 'cyan'))
    
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
    
    if has_prophet:
        print(f"Previsão Prophet: {future_predictions_df['Prophet'].iloc[0]:.4f} (var: {future_predictions_df['Prophet'].iloc[0]-last_price:.4f})")
    
    if has_arima:
        print(f"Previsão ARIMA(2,1,2): {future_predictions_df['ARIMA'].iloc[0]:.4f} (var: {future_predictions_df['ARIMA'].iloc[0]-last_price:.4f})")
    
    if has_ar2:
        print(f"Previsão AR(2): {future_predictions_df['AR(2)'].iloc[0]:.4f} (var: {future_predictions_df['AR(2)'].iloc[0]-last_price:.4f})")
    
    print("\n=== MÉTRICAS DE ERRO (MSE) ===")
    for model, mse in mse_results.items():
        print(f"{model}: {mse:.6f}")
    
    
    sorted_models = sorted(mse_results.items(), key=lambda x: x[1])
    print("\nModelos com menor MSE:")
    for model, mse in sorted_models:
        print(f"{model}: {mse:.6f}")
    

def plot_zoom_graph(df, days_to_show=30, future_days=5):
    """
    Plota um gráfico com zoom nos dados mais recentes e previsões futuras
    
    Args:
        df: DataFrame com os dados históricos
        days_to_show: Número de dias recentes para mostrar
        future_days: Número de dias para projetar no futuro
    """
    # Carregar as previsões dos arquivos CSV
    naive_predictions = pd.read_csv('models_output/predictions_naive.csv', index_col=0, parse_dates=True)
    ma_predictions = pd.read_csv('models_output/predictions_ma.csv', index_col=0, parse_dates=True)
    reg_predictions = pd.read_csv('models_output/predictions_regression.csv', index_col=0, parse_dates=True)
    
    # Verificar se existe o arquivo de previsões dos modelos opcionais
    ses_file = 'models_output/predictions_ses.csv'
    prophet_file = 'models_output/predictions_prophet.csv'
    arima_file = 'models_output/predictions_arima.csv'
    ar2_file = 'models_output/predictions_ar2.csv'
    
    has_ses = os.path.exists(ses_file)
    has_prophet = os.path.exists(prophet_file)
    has_arima = os.path.exists(arima_file)
    has_ar2 = os.path.exists(ar2_file)
    
    if has_ses:
        ses_predictions = pd.read_csv(ses_file, index_col=0, parse_dates=True)
    if has_prophet:
        prophet_predictions = pd.read_csv(prophet_file, index_col=0, parse_dates=True)
    if has_arima:
        arima_predictions = pd.read_csv(arima_file, index_col=0, parse_dates=True)
    if has_ar2:
        ar2_predictions = pd.read_csv(ar2_file, index_col=0, parse_dates=True)
    
    # Calcular MSE para cada modelo
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
    
    # MSE dos outros modelos, se disponíveis
    if has_ses:
        common_dates = df.index.intersection(ses_predictions.index)
        mse_results['SES'] = mean_squared_error(df.loc[common_dates, 'Close'], ses_predictions.loc[common_dates, 'Prediction'])
    
    if has_prophet:
        common_dates = df.index.intersection(prophet_predictions.index)
        mse_results['Prophet'] = mean_squared_error(df.loc[common_dates, 'Close'], prophet_predictions.loc[common_dates, 'Prediction'])
    
    if has_arima:
        common_dates = df.index.intersection(arima_predictions.index)
        mse_results['ARIMA'] = mean_squared_error(df.loc[common_dates, 'Close'], arima_predictions.loc[common_dates, 'Prediction'])
    
    if has_ar2:
        common_dates = df.index.intersection(ar2_predictions.index)
        mse_results['AR(2)'] = mean_squared_error(df.loc[common_dates, 'Close'], ar2_predictions.loc[common_dates, 'Prediction'])
    
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
    
    if has_prophet:
        future_predictions['Prophet'] = [prophet_predictions['Prediction'].iloc[-1]]
    
    if has_arima:
        future_predictions['ARIMA'] = [arima_predictions['Prediction'].iloc[-1]]
    
    if has_ar2:
        future_predictions['AR(2)'] = [ar2_predictions['Prediction'].iloc[-1]]
    
    future_predictions_df = pd.DataFrame(future_predictions).set_index('Date')
    
    # Preparar dados para plotagem (apenas os últimos dias_to_show dias)
    recent_data = df.iloc[-days_to_show:].copy()
    
    # Criar figura
    plt.figure(figsize=(14, 8))
    
    # Plotar dados históricos
    plt.plot(recent_data.index, recent_data['Close'], 
             label='Dados Históricos', color='blue', linewidth=2.5)
    
    # Filtrar apenas dados recentes para o gráfico
    start_date = recent_data.index[0]
    
    # Plotar as linhas de previsão para o histórico
    naive_recent = naive_predictions[naive_predictions.index >= start_date]
    ma_recent = ma_predictions[ma_predictions.index >= start_date]
    reg_recent = reg_predictions[reg_predictions.index >= start_date]
    
    plt.plot(naive_recent.index, naive_recent['Prediction'], 
             label=f'Previsão Naive (MSE: {mse_results["Naive"]:.4f})', 
             color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.plot(ma_recent.index, ma_recent['Prediction'], 
             label=f'Previsão MA(10) (MSE: {mse_results["MA"]:.4f})', 
             color='green', linestyle='--', alpha=0.7, linewidth=2)
    plt.plot(reg_recent.index, reg_recent['Prediction'], 
             label=f'Previsão Regressão (MSE: {mse_results["Regression"]:.4f})', 
             color='magenta', linestyle='--', alpha=0.7, linewidth=2)
    
    # Plotar os modelos opcionais se disponíveis
    if has_ses:
        ses_recent = ses_predictions[ses_predictions.index >= start_date]
        plt.plot(ses_recent.index, ses_recent['Prediction'], 
                label=f'Previsão SES (MSE: {mse_results["SES"]:.4f})', 
                color='orange', linestyle='--', alpha=0.7, linewidth=2)
    
    if has_prophet:
        prophet_recent = prophet_predictions[prophet_predictions.index >= start_date]
        plt.plot(prophet_recent.index, prophet_recent['Prediction'], 
                label=f'Previsão Prophet (MSE: {mse_results["Prophet"]:.4f})', 
                color='purple', linestyle='--', alpha=0.7, linewidth=2)
    
    if has_arima:
        arima_recent = arima_predictions[arima_predictions.index >= start_date]
        plt.plot(arima_recent.index, arima_recent['Prediction'], 
                label=f'Previsão ARIMA(2,1,2) (MSE: {mse_results["ARIMA"]:.4f})', 
                color='brown', linestyle='--', alpha=0.7, linewidth=2)
    
    if has_ar2:
        ar2_recent = ar2_predictions[ar2_predictions.index >= start_date]
        plt.plot(ar2_recent.index, ar2_recent['Prediction'], 
                label=f'Previsão AR(2) (MSE: {mse_results["AR(2)"]:.4f})', 
                color='cyan', linestyle='--', alpha=0.7, linewidth=2)
    
    # Calcular a data do último dia conhecido
    last_date = df.index[-1]
    
    # Adicionar linhas verticais para marcar o último dia com dados reais
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    
    
    # Usar formatação de data para o eixo X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    # Adicionar grade, legenda e títulos
    plt.grid(True, alpha=0.3)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço de Fechamento', fontsize=12)
    plt.title('Zoom nos Dados Recentes e Previsões Futuras - CPTS11', fontsize=14)
    plt.legend(loc='best')
    
    # Rotacionar os rótulos do eixo X
    plt.xticks(rotation=45)
    
    # Adicionar linha horizontal para o último preço conhecido
    last_price = df['Close'].iloc[-1]
    plt.axhline(y=last_price, color='blue', linestyle='--', alpha=0.3)
    plt.annotate(f'Último preço: {last_price:.2f}', 
                xy=(last_date, last_price),
                xytext=(10, -20),
                textcoords='offset points',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    
    # Adicionar anotações com os valores das previsões
    models_colors = [
        ('Naive', 'red'), 
        ('MA', 'green'), 
        ('Regression', 'magenta')
    ]
    
    if has_ses:
        models_colors.append(('SES', 'orange'))
    
    if has_prophet:
        models_colors.append(('Prophet', 'purple'))
    
    if has_arima:
        models_colors.append(('ARIMA', 'brown'))
    
    if has_ar2:
        models_colors.append(('AR(2)', 'cyan'))
    
    for model, color in models_colors:
        pred_value = future_predictions_df[model].iloc[0]
        plt.annotate(f'{model}: {pred_value:.2f}', 
                    xy=(future_predictions_df.index[0], pred_value),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8),
                    color=color)
    
    # Configurar os limites do eixo Y para dar mais espaço para as anotações
    min_price = recent_data['Close'].min() * 0.95
    max_price = recent_data['Close'].max() * 1.05
    plt.ylim(min_price, max_price)
    
    # Adicionar margens para melhor visualização
    plt.tight_layout()
    
    # Salvar gráfico
    output_dir = 'graficos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/zoom_dados_recentes.png', dpi=300, bbox_inches='tight')
    
    # Mostrar o gráfico
    plt.show()

def plot_individual_models(df, days_to_show=300):
    """
    Plota gráficos individuais para cada modelo
    """
    # Carregar as previsões dos arquivos CSV
    models_data = {
        'Naive': ('predictions_naive.csv', 'red'),
        'MA': ('predictions_ma.csv', 'green'),
        'Regression': ('predictions_regression.csv', 'magenta'),
        'SES': ('predictions_ses.csv', 'orange'),
        'Prophet': ('predictions_prophet.csv', 'purple'),
        'ARIMA': ('predictions_arima.csv', 'brown'),
        'AR(2)': ('predictions_ar2.csv', 'cyan')
    }
    
    # Preparar dados para plotagem
    recent_data = df.iloc[-days_to_show:].copy()
    
    for model_name, (filename, color) in models_data.items():
        file_path = f'models_output/{filename}'
        if not os.path.exists(file_path):
            continue
            
        # Carregar previsões do modelo
        predictions = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Calcular MSE
        common_dates = df.index.intersection(predictions.index)
        mse = mean_squared_error(df.loc[common_dates, 'Close'], 
                               predictions.loc[common_dates, 'Prediction'])
        
        # Criar figura individual
        plt.figure(figsize=(12, 6))
        
        # Plotar dados históricos
        plt.plot(recent_data.index, recent_data['Close'], 
                label='Dados Históricos', color='blue', linewidth=2)
        
        # Filtrar dados recentes para o gráfico
        start_date = recent_data.index[0]
        model_recent = predictions[predictions.index >= start_date]
        
        # Plotar previsões do modelo
        plt.plot(model_recent.index, model_recent['Prediction'],
                label=f'Previsão {model_name} (MSE: {mse:.4f})',
                color=color, linestyle='--', alpha=0.7)
        
        # Plotar previsão para o próximo dia
        next_date = predictions.index[-1] + pd.Timedelta(days=1)
        next_pred = predictions['Prediction'].iloc[-1]
        plt.plot(next_date, next_pred, 'o', color=color, markersize=8)
        
        # Adicionar linha vertical para o último dia conhecido
        plt.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.7)
        
        # Configurar eixo X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # Adicionar elementos do gráfico
        plt.grid(True, alpha=0.3)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço de Fechamento', fontsize=12)
        plt.title(f'Modelo {model_name} - CPTS11', fontsize=14)
        plt.legend(loc='best')
        
        # Rotacionar rótulos do eixo X
        plt.xticks(rotation=45)
        
        # Adicionar anotações
        last_price = df['Close'].iloc[-1]
        plt.annotate(f'Último: {last_price:.2f}',
                    xy=(df.index[-1], last_price),
                    xytext=(10, -20),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
        
        plt.annotate(f'Previsão: {next_pred:.2f}',
                    xy=(next_date, next_pred),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8),
                    color=color)
        
        plt.tight_layout()
        
        # Salvar gráfico individual
        output_dir = 'graficos'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/modelo_{model_name.lower().replace("(","").replace(")","")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Carregar dados
    data = load_data()
    print(f"Dados carregados: {data.shape[0]} observações")
    
    # Plotar dados e previsões dos modelos
    plot_all_models(data)
    
    # Plotar gráfico com zoom nos dados recentes
    plot_zoom_graph(data, days_to_show=30, future_days=1)
    
    # Plotar gráficos individuais para cada modelo
    plot_individual_models(data)