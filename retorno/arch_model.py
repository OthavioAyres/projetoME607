import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data, calcular_retorno_log
print("sys.path", sys.path)
if __name__ == "__main__":
    p_modelo = 10
    # Carregar os dados de fechamento
    df = load_data()
    print("Dados carregados:")
    print(df.head())

    # Calcular os retornos logarítmicos
    retornos = calcular_retorno_log(df)
    print("Retornos logarítmicos calculados:")
    print(retornos.head())

    # Ajustar o modelo ARCH(10)
    from arch import arch_model
    modelo = arch_model(retornos, vol='ARCH', p=p_modelo, rescale=False)
    resultado = modelo.fit(disp='off')
    print("\nResumo do ajuste do modelo ARCH(10):")
    print(resultado.summary())

    # Salvar resíduos e volatilidade estimada para uso posterior
    residuos = resultado.resid
    volatilidade = resultado.conditional_volatility

    # Exportar resíduos e volatilidade para arquivos CSV
    residuos.to_csv('residuos_arch10.csv', header=True)
    volatilidade.to_csv('volatilidade_arch10.csv', header=True)

    # Gerar e exibir gráfico da volatilidade estimada ao longo do tempo
    import matplotlib.pyplot as plt
    output_dir = os.path.dirname(os.path.abspath(__file__))

    plt.figure(figsize=(12, 5))
    plt.plot(volatilidade, label='Volatilidade Condicional (ARCH(10))')
    plt.title('Volatilidade Estimada ao Longo do Tempo - ARCH(10)')
    plt.xlabel('Data')
    plt.ylabel('Volatilidade')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volatilidade_arch.png'))
    plt.show()

    # Gerar e exibir gráfico dos resíduos do modelo (e_hat)
    plt.figure(figsize=(12, 5))
    plt.plot(residuos, label='Resíduos (e_hat)', color='orange')
    plt.title('Resíduos do Modelo ARCH(10)')
    plt.xlabel('Data')
    plt.ylabel('Resíduo')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuos_arch.png'))
    plt.show()

    # Gerar e exibir gráfico dos resíduos ao quadrado (e_hat²)
    plt.figure(figsize=(12, 5))
    plt.plot(residuos**2, label='Resíduos ao Quadrado (e_hat²)', color='green')
    plt.title('Resíduos ao Quadrado do Modelo ARCH(10)')
    plt.xlabel('Data')
    plt.ylabel('Resíduo ao Quadrado')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuos2_arch.png'))
    plt.show()

    # Calcular e exibir a ACF dos resíduos (e_hat)
    from statsmodels.graphics.tsaplots import plot_acf
    plt.figure(figsize=(10, 4))
    plot_acf(residuos.dropna(), lags=40, title='ACF dos Resíduos (e_hat)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acf_residuos_arch.png'))
    plt.show()

    # Calcular e exibir a ACF dos resíduos ao quadrado (e_hat²)
    plt.figure(figsize=(10, 4))
    plot_acf((residuos**2).dropna(), lags=40, title='ACF dos Resíduos ao Quadrado (e_hat²)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acf_residuos2_arch.png'))
    plt.show()

    # Aplicar o teste Weighted Ljung-Box nos resíduos ao quadrado
    from statsmodels.stats.diagnostic import acorr_ljungbox
    print('\nTeste Weighted Ljung-Box nos resíduos ao quadrado (e_hat²):')
    lags_to_test = [1, 3*p_modelo-1, 5*p_modelo-1]
    for lag in lags_to_test:
        ljungbox_result = acorr_ljungbox((residuos**2).dropna(), lags=[lag], return_df=True)
        print(f"\nLag = {lag}")
        print(ljungbox_result)
        p_value = ljungbox_result['lb_pvalue'].iloc[0]
        if p_value < 0.05:
            print(f"Resultado: Há evidência de autocorrelação significativa nos resíduos ao quadrado para lag {lag} (p-valor = {p_value:.4f})")
        else:
            print(f"Resultado: Não há evidência de autocorrelação significativa nos resíduos ao quadrado para lag {lag} (p-valor = {p_value:.4f})")


    # Os retornos estão prontos para serem usados no ajuste do modelo ARCH 