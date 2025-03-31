# Projeto de Análise e Previsão da CPTS11

Este projeto implementa diferentes modelos para análise e previsão de séries temporais de preços do fundo imobiliário CPTS11 (Capitânia Securities II FII).

## Descrição

O CPTS11 é um fundo de investimento imobiliário (FII) brasileiro negociado na B3. Este projeto visa analisar o comportamento histórico dos preços e implementar diversos modelos de previsão para comparar suas performances.

## Estrutura do Projeto

```
├── data_utils.py         # Funções para carregamento de dados
├── naive_model.py        # Implementação do modelo Naive
├── moving_average_model.py # Implementação do modelo de Médias Móveis
├── arima_model.py        # Implementação do modelo ARIMA
├── regression_model.py   # Implementação do modelo de Regressão Linear
├── visualization.py      # Funções para visualização
├── main.py               # Script principal para executar todos os modelos
├── CPTS11_historico.csv  # Arquivo de dados históricos
└── resultados/           # Diretório para salvar resultados (criado automaticamente)
```

## Modelos Implementados

### 1. Modelo Naive
O modelo ingênuo (naive) utiliza o último valor observado como previsão para o próximo período. É um baseline simples para comparação com modelos mais complexos.

### 2. Modelo de Médias Móveis
Este modelo calcula a média dos últimos N dias para prever o próximo valor. São testadas diferentes janelas de tempo (5, 10, 20 e 50 dias) para identificar qual produz os melhores resultados.

### 3. Modelo ARIMA
ARIMA (AutoRegressive Integrated Moving Average) é um modelo estatístico que analisa dependências temporais nos dados. O código busca automaticamente os melhores parâmetros (p, d, q) para o modelo.

### 4. Modelo de Regressão Linear
Este modelo utiliza características criadas a partir dos dados históricos (como lags, médias móveis, dia da semana, etc.) para prever os preços futuros através de regressão linear.

## Requisitos

Para executar este projeto, você precisa das seguintes bibliotecas Python:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

Você pode instalar todas as dependências necessárias com:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Como Executar

1. Clone o repositório ou baixe os arquivos
2. Certifique-se de que o arquivo `CPTS11_historico.csv` está no diretório raiz
3. Execute o script principal:

```bash
python main.py
```

Este comando irá:
- Carregar os dados históricos
- Executar todos os modelos implementados
- Comparar seus desempenhos
- Gerar visualizações e salvar os resultados no diretório `resultados/`

## Métricas de Avaliação

Os modelos são avaliados usando as seguintes métricas:
- MAE (Mean Absolute Error) - Erro Absoluto Médio
- RMSE (Root Mean Square Error) - Erro Quadrático Médio
- R² (Coeficiente de Determinação)
- MAPE (Mean Absolute Percentage Error) - Erro Percentual Absoluto Médio

## Arquivos de Resultado

Após a execução, serão gerados os seguintes arquivos de resultado:
- `resultados/comparacao_modelos.png`: Gráfico comparando o desempenho dos modelos
- `resultados/previsoes_futuras.png`: Gráfico com as previsões futuras dos modelos
- `resultados/avaliacao_modelos.csv`: Tabela com as métricas de avaliação de cada modelo

Além disso, cada modelo individual também gera suas próprias visualizações específicas.

## Autor

Projeto desenvolvido para disciplina ME607 - Séries Temporais, UNICAMP. 