# Churn Model — Previsão de Churn de Clientes (Telco)

Modelo preditivo para identificar clientes com alta probabilidade de cancelamento (*churn*), construído sobre o dataset **Telco Customer Churn**. O projeto cobre o pipeline completo: EDA, pré-processamento, tratamento de desbalanceamento, seleção e otimização de modelo, ajuste de threshold de decisão, interpretabilidade, segmentação de clientes (clustering) e entrega de artefatos prontos para inferência.

## Conteúdo do repositório

| Arquivo/Pasta | Descrição |
|---|---|
| `churn_notebook.ipynb` | Notebook principal com o pipeline completo de modelagem (EDA → deploy), dividido em 13 fases. |
| `apresentacao_cliente_churn.ipynb` | Notebook de apresentação dos resultados, voltado para leitura por stakeholders. |
| `TelcoCustomerChurn.csv` | Dataset bruto utilizado no treinamento. |
| `dicionario_dados_churn.xlsx` | Dicionário de dados com a descrição de cada variável. |
| `artefatos_modelo/pipeline_churn.pkl` | Pipeline treinado (pré-processamento + modelo) para previsão de churn. |
| `artefatos_modelo/pipeline_clusters.pkl` | Pipeline de clustering para segmentação de clientes. |
| `artefatos_modelo/metadados.json` | Metadados do modelo: threshold de decisão, features esperadas e métricas de teste. |
| `artefatos_modelo/metadados_clusters.json` | Perfis e métricas dos clusters de clientes identificados. |
| `artefatos_modelo/predict.py` | Script de inferência em produção. |
| `artefatos_modelo/requirements.txt` | Dependências necessárias para rodar o modelo. |

## Pipeline do notebook (`churn_notebook.ipynb`)

1. Carregamento e inspeção inicial dos dados
2. Análise exploratória (target, numéricas, categóricas, nulos, outliers, separabilidade, correlações, dimensão temporal)
3. Pré-processamento (incluindo correção de colinearidade perfeita)
4. Divisão treino/teste
5. Tratamento de desbalanceamento de classes
6. Escalonamento
7. Seleção e treinamento de modelos base
8. Otimização de hiperparâmetros
9. Ajuste de threshold de decisão
10. Diagnóstico e testes de qualidade
11. Interpretabilidade (incluindo análise dos erros mais "confiantes")
12. Segmentação de clientes via clustering
13. Persistência dos artefatos finais

## Resultados do modelo

Classe positiva: `Churn = Yes`. O threshold de decisão foi ajustado para priorizar recall, assumindo que um falso negativo (cliente que cancela e não foi identificado) é mais custoso que um falso positivo.

| Métrica | Valor |
|---|---|
| Threshold de decisão | 0.15 |
| Accuracy | 0.6593 |
| Precision | 0.4329 |
| Recall | 0.9144 |
| F1 | 0.5876 |
| ROC AUC | 0.8453 |
| PR AUC | 0.6623 |

## Segmentação de clientes (clustering)

O modelo de clustering (k=4) identificou os seguintes perfis de clientes:

| Cluster | Nome | % da base | Taxa de churn | Perfil |
|---|---|---|---|---|
| 0 | Só Telefone, Estáveis | 21.7% | 7% | Sem internet, mensalidade baixa, churn muito baixo. Sem prioridade de retenção. |
| 1 | Premium Fidelizados | 28.7% | 13% | Maior tenure e gasto histórico, muitos serviços adicionais. Clientes de maior valor — proteger ativamente. |
| 2 | DSL em Transição | 23.4% | 26% | Tenure baixo-médio, maioria em contrato mensal. Ainda "decidindo" se ficam. |
| 3 | Fibra de Alto Risco | 26.2% | 57% | Fibra óptica, contrato mensal, menor tenure, mensalidade alta. Prioridade máxima de retenção. |

## Como usar

### Instalação

```bash
pip install -r artefatos_modelo/requirements.txt
```

### Rodando uma previsão

```bash
python artefatos_modelo/predict.py caminho_para_novos_clientes.csv
```

O script espera um CSV com as mesmas colunas do dataset original (`TelcoCustomerChurn.csv`) e gera `previsoes_churn.csv` com a probabilidade e a previsão de churn (0/1) para cada cliente, com base no threshold definido em `metadados.json`.

### Features esperadas como input

`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` (as duas últimas — `num_servicos_adicionais` e `cobranca_media_mensal_historica` — são derivadas automaticamente pelo script).

## Premissas de negócio

- Classe positiva: `Churn = Yes`.
- Custo assumido: falso negativo mais caro que falso positivo (por isso o threshold foi reduzido para priorizar recall).
- Caso o custo real de negócio seja diferente do assumido, o `THRESHOLD_FINAL` deve ser reajustado.
