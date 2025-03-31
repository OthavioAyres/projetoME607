
# Projeto de Análise e Previsão da CPTS11

Este projeto implementa diferentes modelos para análise e previsão de séries temporais de preços do fundo imobiliário CPTS11 (Capitânia Securities II FII).

## Descrição

O CPTS11 é um fundo de investimento imobiliário (FII) brasileiro negociado na B3. Este projeto visa analisar o comportamento histórico dos preços e implementar diversos modelos de previsão para comparar suas performances.

## Estrutura do Projeto

```
├── data_utils.py         # Funções para carregamento de dados
├── models/               # Pasta com implementações dos modelos
│   ├── __init__.py       # Arquivo de inicialização do pacote
│   ├── naive_model.py    # Implementação do modelo Naive
│   ├── moving_average_model.py # Implementação do modelo de Médias Móveis
│   ├── regression_model.py     # Implementação do modelo de Regressão Linear
├── models_output/        # Diretório para arquivos CSV gerados pelos modelos
│   ├── predictions_naive.csv       # Previsões do modelo Naive
│   ├── predictions_ma.csv          # Previsões do modelo de Médias Móveis
│   ├── predictions_regression.csv  # Previsões do modelo de Regressão Linear
├── main.py               # Script principal para visualização e comparação
├── CPTS11_historico.csv  # Arquivo de dados históricos
├── graficos/             # Diretório para salvar gráficos (criado automaticamente)
```

## Modelos Implementados

### 1. Modelo Naive
O modelo ingênuo (naive) utiliza o último valor observado como previsão para o próximo período. É um baseline simples para comparação com modelos mais complexos.

### 2. Modelo de Médias Móveis
Este modelo calcula a média dos últimos 10 dias para prever o próximo valor. A média móvel suaviza as flutuações de curto prazo e destaca tendências de médio prazo.

### 3. Modelo de Regressão Linear
Este modelo utiliza uma regressão linear simples sobre todos os dados históricos para capturar a tendência geral dos preços. A função da linha de tendência é então utilizada para prever o próximo dia.

## Fluxo de Execução

O projeto segue um fluxo em duas etapas:

1. **Geração de previsões**: Cada modelo gera um arquivo CSV com suas previsões para todos os dias históricos e para o próximo dia.
2. **Visualização e comparação**: O script principal (`main.py`) lê os arquivos CSV e gera gráficos comparativos.

## Requisitos

Para executar este projeto, você precisa das seguintes bibliotecas Python:
- pandas
- numpy
- matplotlib
- scikit-learn

Você pode instalar todas as dependências necessárias com:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Como Executar

1. Clone o repositório ou baixe os arquivos
2. Certifique-se de que o arquivo `CPTS11_historico.csv` está no diretório raiz
3. Execute primeiro cada modelo para gerar as previsões:

```bash
python -m models.naive_model
python -m models.moving_average_model
python -m models.regression_model
```

4. Em seguida, execute o script principal para visualizar a comparação:

```bash
python main.py
```

## Métricas de Avaliação

Os modelos são avaliados usando o MSE (Mean Squared Error - Erro Quadrático Médio). Esta métrica é calculada comparando as previsões históricas com os valores reais observados.

## Visualizações Geradas

Após a execução, serão gerados os seguintes gráficos:
- `graficos/comparacao_modelos.png`: Gráfico comparando os modelos com:
  - Dados históricos
  - Linhas de previsão para cada modelo
  - Valores MSE para cada modelo
  - Previsões pontuais para o próximo dia
  - Anotações explicativas

A visualização inclui:
- Linhas de previsão históricas para cada modelo
- Pontos de previsão para o próximo dia
- Valores MSE para comparação quantitativa
- Identificação automática do melhor modelo (menor MSE)

## Autor

Projeto desenvolvido para disciplina ME607 - Séries Temporais, UNICAMP.
