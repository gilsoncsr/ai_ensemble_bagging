# 📊 Lead Conversion Analysis and Prediction

This project aims to analyze a dataset of **sales leads** and build a predictive model to identify which leads are more likely to convert. The analysis involves **data cleaning**, **exploratory data analysis (EDA)**, **feature engineering**, and **machine learning modeling** using `Bagging` with `Logistic Regression`.

---

## 🔧 Technologies and Libraries Used

- **Python 3.11+**
- **Pandas** — data manipulation and cleaning
- **NumPy** — mathematical operations
- **Matplotlib, Seaborn, Plotly** — data visualization
- **Scikit-learn** — preprocessing, modeling, and model evaluation
- **SciPy** — statistical tests (chi-square)

---

## 📂 Project Structure

```
.
├── datasets/
│   └── leads.csv
├── lead_conversion_analysis.ipynb
└── README.md
```

---

## 📈 Project Steps

### 1. Data Loading and Initial Visualization

- Displaying the first and last rows.
- Structure analysis using `.info()`.

### 2. Data Cleaning and Feature Engineering

- Removing irrelevant or redundant columns.
- Standardizing values (e.g., "google" → "Google").
- Converting Yes/No columns to binary values.
- Removing columns with excessive missing values (> 25%).

### 3. Exploratory Data Analysis (EDA)

- Calculating and visualizing the **Hit Ratio**.
- Generating **boxplots** for numerical variables vs. target.
- Correlation analysis between variables.
- Chi-square independence tests between categorical variables and the target.

### 4. Data Preparation for Modeling

- Splitting into independent variables (X) and target (y).
- Preprocessing with `ColumnTransformer` for normalization and one-hot encoding.
- Splitting into training and test sets.

### 5. Predictive Modeling

- Training a `BaggingClassifier` model based on `LogisticRegression`.

### 6. Model Evaluation

- Metrics used: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**.

---

## 🧪 Results and Insights

- Statistical analysis revealed that **Lead Source**, **Lead Origin**, and **Last Notable Activity** have significant impact on conversion.
- The trained model can be used to prioritize leads with a higher probability of conversion, optimizing marketing and sales efforts.

---

## 🚀 How to Run

1. Clone this repository.
2. Create a virtual environment using `pipenv` or `venv`.

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the `lead_conversion_analysis.ipynb` notebook in a Jupyter environment.

---

## 📌 Notes

- The dataset used is available at `./datasets/leads.csv`.
- This is an educational project and can be expanded with more advanced techniques such as Random Forest, XGBoost, or SHAP values analysis.

---

# 📊 Análisis y Predicción de Conversión de Leads

Este proyecto tiene como objetivo analizar un conjunto de datos de **leads de ventas** y construir un modelo predictivo para identificar qué leads tienen mayor probabilidad de conversión. El análisis incluye **limpieza de datos**, **análisis exploratorio de datos (EDA)**, **ingeniería de características**, y **modelado con aprendizaje automático** utilizando `Bagging` con `Regresión Logística`.

---

## 🔧 Tecnologías y Bibliotecas Utilizadas

- **Python 3.11+**
- **Pandas** — manipulación y limpieza de datos
- **NumPy** — operaciones matemáticas
- **Matplotlib, Seaborn, Plotly** — visualización de datos
- **Scikit-learn** — preprocesamiento, modelado y evaluación de modelos
- **SciPy** — pruebas estadísticas (chi-cuadrado)

---

## 📂 Estructura del Proyecto

```
.
├── datasets/
│   └── leads.csv
├── lead_conversion_analysis.ipynb
└── README.md
```

---

## 📈 Etapas del Proyecto

### 1. Carga de Datos y Visualización Inicial

- Visualización de las primeras y últimas filas.
- Análisis de la estructura con `.info()`.

### 2. Limpieza de Datos e Ingeniería de Características

- Eliminación de columnas irrelevantes o redundantes.
- Estandarización de valores (ej.: "google" → "Google").
- Conversión de columnas Sí/No a valores binarios.
- Eliminación de columnas con valores faltantes excesivos (> 25%).

### 3. Análisis Exploratorio de Datos (EDA)

- Cálculo y visualización del **Hit Ratio**.
- Generación de **boxplots** para variables numéricas vs. la variable objetivo.
- Análisis de correlación entre variables.
- Pruebas de independencia chi-cuadrado entre variables categóricas y la variable objetivo.

### 4. Preparación de los Datos para el Modelo

- Separación en variables independientes (X) y dependiente (y).
- Preprocesamiento con `ColumnTransformer` para normalización y codificación one-hot.
- División en conjuntos de entrenamiento y prueba.

### 5. Modelado Predictivo

- Entrenamiento de un modelo `BaggingClassifier` basado en `LogisticRegression`.

### 6. Evaluación del Modelo

- Métricas utilizadas: **Accuracy**, **Precision**, **Recall**, **F1-score** y **Matriz de Confusión**.

---

## 🧪 Resultados e Insights

- El análisis estadístico reveló que **Lead Source**, **Lead Origin** y **Last Notable Activity** tienen un impacto significativo en la conversión.
- El modelo entrenado puede usarse para priorizar leads con mayor probabilidad de conversión, optimizando los esfuerzos de marketing y ventas.

---

## 🚀 Cómo Ejecutar

1. Clona este repositorio.
2. Crea un entorno virtual usando `pipenv` o `venv`.

3. Instala las dependencias:
   ```bash
   pipenv install pandas numpy matplotlib seaborn plotly nbformat scipy scikit-learn ipykernel
   ```
4. Ejecuta el notebook `lead_conversion_analysis.ipynb` en un entorno Jupyter.

---

## 📌 Notas

- El conjunto de datos utilizado está disponible en `./datasets/leads.csv`.
- Este es un proyecto educativo y puede ampliarse con técnicas más avanzadas como Random Forest, XGBoost o análisis de valores SHAP.

---

# 📊 Análise e Predição de Conversão de Leads

Este projeto tem como objetivo analisar um conjunto de dados de **leads de vendas** e construir um modelo preditivo para identificar quais leads têm maior probabilidade de conversão. A análise envolve **limpeza de dados**, **análise exploratória (EDA)**, **engenharia de atributos**, e **modelagem com aprendizado de máquina** utilizando `Bagging` com `Regressão Logística`.

---

## 🔧 Tecnologias e Bibliotecas Utilizadas

- **Python 3.11+**
- **Pandas** — manipulação e limpeza de dados
- **NumPy** — operações matemáticas
- **Matplotlib, Seaborn, Plotly** — visualização de dados
- **Scikit-learn** — pré-processamento, modelagem e avaliação de modelos
- **SciPy** — testes estatísticos (qui-quadrado)

---

## 📂 Estrutura do Projeto

```
.
├── datasets/
│   └── leads.csv
├── lead_conversion_analysis.ipynb
└── README.md
```

---

## 📈 Etapas do Projeto

### 1. Carregamento e Visualização Inicial dos Dados

- Visualização das primeiras e últimas linhas.
- Análise de estrutura com `.info()`.

### 2. Limpeza e Engenharia de Atributos

- Remoção de colunas irrelevantes ou redundantes.
- Padronização de valores (ex: "google" → "Google").
- Conversão de colunas Yes/No para valores binários.
- Remoção de colunas com valores ausentes excessivos (> 25%).

### 3. Análise Exploratória (EDA)

- Cálculo e visualização do **Hit Ratio**.
- Geração de **boxplots** para variáveis numéricas em relação ao target.
- Análise de correlação entre variáveis.
- Testes de independência qui-quadrado entre variáveis categóricas e a variável alvo.

### 4. Preparação dos Dados para o Modelo

- Separação entre variáveis independentes (X) e dependente (y).
- Pré-processamento com `ColumnTransformer` para normalização e codificação one-hot.
- Divisão em conjuntos de treino e teste.

### 5. Modelagem Preditiva

- Treinamento de um modelo `BaggingClassifier` com base em `LogisticRegression`.

### 6. Avaliação do Modelo

- Métricas utilizadas: **Accuracy**, **Precision**, **Recall**, **F1-score** e **Matriz de Confusão**.

---

## 🧪 Resultados e Insights

- A análise estatística revelou que **Lead Source**, **Lead Origin** e **Last Notable Activity** têm impacto significativo na conversão.
- O modelo treinado pode ser usado para priorizar leads com maior probabilidade de conversão, otimizando esforços de marketing e vendas.

---

## 🚀 Como Executar

1. Clone este repositório.
2. Crie um ambiente virtual com `pipenv` ou `venv`.

3. Instale as dependências:
   ```bash
   pipenv install pandas numpy matplotlib seaborn plotly nbformat scipy scikit-learn ipykernel
   ```
4. Execute o notebook `lead_conversion_analysis.ipynb` em um ambiente Jupyter.

---

## 📌 Observações

- Os dados utilizados estão disponíveis em `./datasets/leads.csv`.
- Este projeto é didático e pode ser expandido com técnicas mais avançadas como Random Forest, XGBoost ou análise de SHAP values.

---
