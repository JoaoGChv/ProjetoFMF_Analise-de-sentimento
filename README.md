# Projeto FMF vs ORIGINAL MELHORADO: Análise de Sentimentos IMDB

## Visão Geral

Este projeto implementa uma análise completa de sentimentos em reviews de filmes do dataset IMDB, seguindo a estrutura metodológica **AGEMC** (Aplicação, Geração/Coleta, Exploração, Modelagem, Comunicação). 

O projeto foi desenvolvido em duas versões:
- **Projeto FMF**: Implementação fiel ao artigo de referência
- **Projeto ORIGINAL MELHORADO**: Versão com 8 originalidades inovadoras, incluindo validação estatística robusta

### Objetivo
Identificar qual modelo de machine learning é mais eficaz para prever o sentimento (positivo ou negativo) de um review de filme, comparando abordagens tradicionais com inovações em modelagem e comunicação, com validação científica robusta.

---

## Estrutura do Projeto Organizada

```
├── data
│   └── IMDB Dataset.csv
├── docs
│   └── relatorio_final_consolidado.md
├── .git
│   ├── branches
│   ├── COMMIT_EDITMSG
│   ├── config
│   ├── description
│   ├── FETCH_HEAD
│   ├── HEAD
│   ├── hooks
│   │   ├── applypatch-msg.sample
│   │   ├── commit-msg.sample
│   │   ├── fsmonitor-watchman.sample
│   │   ├── post-update.sample
│   │   ├── pre-applypatch.sample
│   │   ├── pre-commit.sample
│   │   ├── pre-merge-commit.sample
│   │   ├── prepare-commit-msg.sample
│   │   ├── pre-push.sample
│   │   ├── pre-rebase.sample
│   │   ├── pre-receive.sample
│   │   ├── push-to-checkout.sample
│   │   ├── sendemail-validate.sample
│   │   └── update.sample
│   ├── index
│   ├── info
│   │   └── exclude
│   ├── logs
│   │   ├── HEAD
│   │   └── refs
│   │       ├── heads
│   │       │   └── main
│   │       └── remotes
│   │           └── origin
│   │               ├── HEAD
│   │               └── main
│   ├── objects
│   │   ├── 01
│   │   │   └── c01b745f06ab3d88d1c7160eb4be55f8547124
│   │   ├── 03
│   │   │   └── 241355775c2a41045ab100c44a79008ef4a4df
│   │   ├── 0c
│   │   │   └── de0738e1c708623e464c351e66205b60ca9f4f
│   │   ├── 1c
│   │   │   └── 9d7ece43fa7a5401e58325b56502772775ffa0
│   │   ├── 27
│   │   │   └── f508d6a9aa7c14987134ee3718100e97e429d6
│   │   ├── 31
│   │   │   └── 1d270ae3219ba2a3a37c834ad0c828b603d050
│   │   ├── 38
│   │   │   └── 5f31dc5cf418bd082b9ffa7a2ba9f80d1b262a
│   │   ├── 44
│   │   │   └── c02116027d73106b6d9aa79b67884b19d01482
│   │   ├── 50
│   │   │   └── 961f643978d63ab979cb1806401e45e5f21ea0
│   │   ├── 5d
│   │   │   └── 5a6571235ab8f0329aad2d7d88a57bf244e13e
│   │   ├── 5f
│   │   │   └── 6ee0714ca18c6f46de3a271117a65ac31c500b
│   │   ├── 63
│   │   │   └── 3f64fcf80ec15e7e281c7a163839282e3f04a5
│   │   ├── 68
│   │   │   └── 52f76dee12a38cbfcc07d166c41103ec13f1b4
│   │   ├── 6b
│   │   │   └── 182d2614a3e1341b6023be6d5b8e5a8bd8675b
│   │   ├── 7c
│   │   │   └── d753b3e5a3f542f32e153b75fdfef342895d00
│   │   ├── a5
│   │   │   └── 9c4232a4b4c670e34388da319459f0c66bf42f
│   │   ├── ae
│   │   │   └── 9a592df1a479ea5a637aabf2ad6ded346c4e49
│   │   ├── bb
│   │   │   └── ebbc4ef3abedc0f65e346c687f7299061aac77
│   │   ├── da
│   │   │   └── 8a80e7d0286ff7503ff96f73f24cd41be1318a
│   │   ├── e1
│   │   │   └── 4928c1b842cf801f78ce6174da3b0698cdf687
│   │   ├── ec
│   │   │   └── 7317514c95427966449a543717f97d272c9aab
│   │   ├── f9
│   │   │   └── c84946626cc8f97be871b1dc5b7bd0a4bdc071
│   │   ├── info
│   │   └── pack
│   │       ├── pack-beb63ddfdafa7d664807f3b09a94906cf90fe915.idx
│   │       ├── pack-beb63ddfdafa7d664807f3b09a94906cf90fe915.pack
│   │       └── pack-beb63ddfdafa7d664807f3b09a94906cf90fe915.rev
│   ├── ORIG_HEAD
│   ├── packed-refs
│   └── refs
│       ├── heads
│       │   └── main
│       ├── remotes
│       │   └── origin
│       │       ├── HEAD
│       │       └── main
│       └── tags
├── .gitignore
├── notebooks
│   └── Projeto_Pad_Original_vs_FMF.ipynb
├── README.md
├── requirements.txt
├── results
│   ├── resultados_modelagem.txt
│   └── resultados_projeto_original.txt
├── tmp_Henrique
│   ├── AGEMC.jpg
│   ├── data
│   │   └── IMDB Dataset.csv
│   ├── docs
│   │   ├── README.md
│   │   ├── relatorio_final_consolidado.md
│   │   └── roteiro_apresentacao.md
│   ├── notebooks
│   │   └── Projeto_Pad_Original_vs_FMF.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── results
│   │   ├── resultados_modelagem.txt
│   │   └── resultados_projeto_original.txt
│   ├── src
│   │   ├── analise_sentimentos_imdb.py
│   │   ├── comunicacao_avancada_imdb.py
│   │   ├── estatisticas_detalhadas.py
│   │   ├── execucao_completa.py
│   │   ├── modelagem_comunicacao_imdb.py
│   │   └── projeto_original_imdb.py
│   └── visualizations
│       ├── analise_features_importancia.png
│       ├── distribuicao_sentimentos.png
│       ├── estatisticas_detalhadas.png
│       ├── frequencia_palavras_por_classe.png
│       ├── matriz_confusao_avancada.png
│       └── wordclouds_analise.png
├── tmp_joão
│   ├── AGEMC.jpg
│   └── Projeto de ML (Seguindo a Documentação) - PAD (JoãoG.).ipynb
└── visualizations
    ├── analise_features_importancia.png
    ├── distribuicao_sentimentos.png
    ├── estatisticas_detalhadas.png
    ├── frequencia_palavras_por_classe.png
    ├── matriz_confusao_avancada.png
    └── wordclouds_analise.png
```

---

## Como Executar

### Importante: Dataset IMDB

O arquivo `data/IMDB Dataset.csv` (63MB) pode ser muito grande para o GitHub. Se você não conseguir fazer o download ou se o arquivo não estiver disponível:

#### Opção 1: Download Manual
1. Acesse: [IMDB Dataset Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Baixe o arquivo `IMDB Dataset.csv`
3. Coloque-o na pasta `data/` do projeto

#### Opção 2: Usar Dataset Menor
Para testes rápidos, você pode usar apenas uma amostra do dataset:
```python
# No código, adicione:
df = df.sample(n=5000, random_state=42)  # Usar apenas 5000 reviews
```

### 1. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 2. Execução Completa (Recomendado)
```bash
python src/execucao_completa.py
```

## Resultados Finais Melhorados

### Comparação de Modelos com Validação Estatística

| Modelo | CV Acurácia | CV F1-Score | Teste Acurácia | Teste F1-Score | Significância |
|--------|-------------|-------------|----------------|----------------|---------------|
| **SVM (Otimizado)** | **83.45%** | **83.71%** | **83.48%** | **83.76%** | **Significativo** |
| Logistic Regression | 83.30% | 83.00% | 83.33% | 83.02% | Não significativo |
| SVM | 82.70% | 83.15% | 82.73% | 83.19% | Significativo |
| Naive Bayes | 81.80% | 80.93% | 81.82% | 80.95% | Não significativo |
| Random Forest | 81.18% | 80.77% | 81.21% | 80.80% | Significativo |
| Decision Tree | 70.25% | 69.80% | 70.30% | 69.85% | Não significativo |

### Campeão Geral
**SVM Otimizado** com 83.48% de acurácia e 83.76% de F1-Score, com validação estatística confirmando significância (p < 0.05)

---

## Originalidades Implementadas (8 total)

### **Projeto ORIGINAL MELHORADO - Fase M (Modelagem)**

#### 1. Pipeline Avançado de Pré-processamento
- Conversão para minúsculas
- Remoção de números e pontuação
- Remoção de stopwords
- Remoção de palavras < 3 letras
- **Lemmatização** (diferencial)

#### 2. Novos Modelos
- **SVM (SVC)**: 82.73% acurácia, 83.19% F1-Score
- **Random Forest**: 81.21% acurácia, 80.80% F1-Score

#### 3. Otimização de Hiperparâmetros
- **GridSearchCV** aplicado ao melhor modelo
- **Melhores parâmetros**: C=10, kernel='rbf', gamma='scale'
- **Melhoria**: +0.75% acurácia e +0.57% F1-Score

#### 4. **VALIDAÇÃO ESTATÍSTICA (NOVA)**
- **Teste t para amostras independentes**
- **Comparação de modelos com p-value**
- **Confirmação de significância estatística**
- **Resultado**: SVM vs Logistic Regression: p < 0.05 (significativo)

#### 5. **CROSS-VALIDATION ROBUSTA (NOVA)**
- **StratifiedKFold** com 5 folds
- **Shuffle=True** para melhor generalização
- **Métricas com média ± desvio padrão**
- **Resultado**: Validação mais confiável

#### 6. **ANÁLISE DE FEATURES (NOVA)**
- **Feature Importances** para Random Forest
- **Coefficient Magnitudes** para Logistic Regression
- **Top 20 features** mais importantes
- **Visualização profissional** das importâncias

### **Projeto ORIGINAL MELHORADO - Fase C (Comunicação)**

#### 7. Visualização Profissional da Matriz de Confusão
- Heatmap com seaborn
- Rótulos claros para TN, FP, FN, TP
- Métricas detalhadas (precisão, recall, especificidade)
- Arquivo: `matriz_confusao_avancada.png`

#### 8. Análise Detalhada de Erros do Modelo
- **Falsos Positivos**: 2 exemplos com análise
- **Falsos Negativos**: 2 exemplos com análise
- Identificação de causas dos erros

---

## Fases AGEMC Implementadas

### Fase A - Aplicação (Problema de Negócio)
**Pergunta Principal**: Qual modelo de machine learning é mais eficaz para prever o sentimento (positivo ou negativo) de um review de filme?

### Fase G - Geração/Coleta de Dados
- **Dataset carregado**: 50.000 reviews
- **Estrutura**: 2 colunas (review, sentiment)
- **Qualidade**: Sem dados faltantes
- **Balanceamento**: 50% positivos, 50% negativos

### Fase E - Exploração (Análise Exploratória)
- **Estatísticas completas** do dataset
- **Visualizações profissionais** (distribuição, word clouds, comprimento)
- **Análise de qualidade** dos dados
- **Insights valiosos** para modelagem

### Fase M - Modelagem
- **Projeto FMF**: Modelos do artigo (Logistic Regression, Naive Bayes, Decision Tree)
- **Projeto ORIGINAL MELHORADO**: Novos modelos + otimização + validação estatística
- **Comparação completa** de performance com significância estatística

### Fase C - Comunicação
- **Projeto FMF**: Métricas básicas e matriz de confusão
- **Projeto ORIGINAL MELHORADO**: Visualizações avançadas, análise de erros e análise de features
- **Relatórios profissionais** para apresentação

---

## Visualizações Geradas

### Análise Exploratória
- `distribuicao_sentimentos.png` - Balanceamento das classes
- `wordclouds_analise.png` - Nuvens de palavras por sentimento
- `analise_comprimento.png` - Distribuição de comprimento dos reviews
- `estatisticas_detalhadas.png` - Estatísticas completas

### Modelagem e Comunicação
- `matriz_confusao.png` - Matriz de confusão do Projeto FMF
- `matriz_confusao_avancada.png` - Matriz de confusão profissional do ORIGINAL MELHORADO
- `frequencia_palavras_por_classe.png` - Importância das palavras
- `analise_features_importancia.png` - Análise de features

---

