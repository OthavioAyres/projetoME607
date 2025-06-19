import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_data, calcular_retorno_log

if __name__ == "__main__":
    df = load_data()
    retornos = calcular_retorno_log(df)
    import pandas as pd
    from scipy.stats import kurtosis

    # retornos é uma série de retornos (porcentuais ou log-retornos)
    print(kurtosis(retornos, fisher=False))  # False = curtose padrão (não excessiva)


    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import numpy as np

    sns.histplot(retornos, kde=True, stat="density", bins=50, label="Retornos")
    x = np.linspace(retornos.min(), retornos.max(), 1000)
    plt.plot(x, stats.norm.pdf(x, retornos.mean(), retornos.std()), label="Normal", color='red')
    plt.legend()
    plt.title("Distribuição dos Retornos")
    plt.savefig("retornos_distribuicao.png")
    plt.show()
    ## salvar o gráfico

