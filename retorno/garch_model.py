import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_data, calcular_retorno_log

if __name__ == "__main__":
    # Carregar os dados de fechamento
    df = load_data()
    print("Dados carregados:")
    print(df.head())

    # Calcular os retornos logarítmicos
    retornos = calcular_retorno_log(df)
    print("Retornos logarítmicos calculados:")
    print(retornos.head())

    # Ajustar o modelo GARCH(1,1)
    from arch import arch_model
    modelo = arch_model(
        retornos,
        vol='GARCH',
        p=2,
        o=0,        # <<< ativa o termo de assimetria!
        q=4,
        dist='t',
        rescale= True
    )
    resultado = modelo.fit(disp='on', options={'maxiter': 100000})
    print("\nResumo do ajuste do modelo GARCH(1,1):")
    print(resultado.summary())
    #print(dir(resultado))
    print(resultado.convergence_flag)
    # Salvar resíduos e volatilidade estimada para uso posterior
    residuos = resultado.resid
    volatilidade = resultado.conditional_volatility
    e_hat = residuos/volatilidade

    # Exportar resíduos e volatilidade para arquivos CSV
    residuos.to_csv('residuos_garch11.csv', header=True)
    volatilidade.to_csv('volatilidade_garch11.csv', header=True)

    # Aplicar o teste Weighted Ljung-Box nos resíduos padronizados ao quadrado (e_hat²):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    print('\nTeste Weighted Ljung-Box nos resíduos padronizados ao quadrado (e_hat²):')
    lags_to_test = [1, 4, 10]
    for lag in lags_to_test:
        ljungbox_result = acorr_ljungbox((e_hat**2).dropna(), lags=[lag], return_df=True)
        print(f"\nLag = {lag}")
        print(ljungbox_result)
        p_value = ljungbox_result['lb_pvalue'].iloc[0]
        # if p_value < 0.05:
        #     print(f"Resultado: Há evidência de autocorrelação significativa nos resíduos ao quadrado para lag {lag} (p-valor = {p_value:.4f})")
        # else:
        #     print(f"Resultado: Não há evidência de autocorrelação significativa nos resíduos ao quadrado para lag {lag} (p-valor = {p_value:.4f})")



    # Gerar e exibir gráficos
    import matplotlib.pyplot as plt
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Gráfico da volatilidade condicional
    plt.figure(figsize=(12, 5))
    plt.plot(volatilidade, label='Volatilidade Condicional (GARCH(1,1))')
    plt.title('Volatilidade Estimada ao Longo do Tempo - GARCH(1,1)')
    plt.xlabel('Data')
    plt.ylabel('Volatilidade')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volatilidade_garch.png'))
    plt.show()

    # Gráfico dos resíduos
    plt.figure(figsize=(12, 5))
    plt.plot(e_hat, label='Resíduos padronizados (e_hat)', color='orange')
    plt.title('Resíduos do Modelo GARCH(1,1)')
    plt.xlabel('Data')
    plt.ylabel('Resíduo')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuos_garch.png'))
    plt.show()

    # Gráfico dos resíduos ao quadrado
    plt.figure(figsize=(12, 5))
    plt.plot(e_hat**2, label='Resíduos padronizados ao Quadrado (e_hat²)', color='green')
    plt.title('Resíduos padronizados ao Quadrado do Modelo GARCH(1,1)')
    plt.xlabel('Data')
    plt.ylabel('Resíduo padronizado ao Quadrado')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuos2_garch.png'))
    plt.show()

    # Calcular e exibir a ACF dos resíduos (e_hat)
    from statsmodels.graphics.tsaplots import plot_acf
    plt.figure(figsize=(10, 4))
    plot_acf(e_hat.dropna(), lags=40, title='ACF dos erros padronizados (e_hat)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acf_residuos_garch.png'))
    plt.show()

    # Calcular e exibir a ACF dos resíduos ao quadrado (e_hat²)
    plt.figure(figsize=(10, 4))
    plot_acf((e_hat**2).dropna(), lags=40, title='ACF dos erros padronizados ao Quadrado (e_hat²)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acf_residuos2_garch.png'))
    plt.show()

    # Grafico de qq plot
    import statsmodels.api as sm
    import scipy.stats as stats
    sm.qqplot(e_hat, dist=stats.t, line='45', fit=True, ax=plt.gca())
    plt.title('GARCH(1,1) - Distribuição t-Student')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qqplot_residuos_garch.png'))
    plt.show()

    # Calcular e plotar a News Impact Curve
    import numpy as np

    def news_impact_curve_garch11(omega, alpha, beta):
        eps = np.linspace(-4, 4, 100)
        var_nc = omega / (1 - alpha - beta)
        nic = omega + alpha * (eps**2) + beta * var_nc
        return eps, nic

    # Obter os parâmetros estimados do modelo
    omega = resultado.params['omega']
    alpha = resultado.params['alpha[1]']
    beta = resultado.params['beta[1]']

    # Calcular a curva de impacto
    eps, nic = news_impact_curve_garch11(omega, alpha, beta)

    # Plotar a curva de impacto
    plt.figure(figsize=(10, 6))
    plt.plot(eps, nic, 'b-', label='News Impact Curve')
    plt.title('Curva de Impacto de Notícias - GARCH(1,1)')
    plt.xlabel('ε (Choque/Notícia)')
    plt.ylabel('σ² (Volatilidade)')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.2)
    plt.axhline(y=omega/(1-alpha-beta), color='g', linestyle='--', alpha=0.2, 
                label='Variância não condicional')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'news_impact_curve_garch.png'))
    plt.show()