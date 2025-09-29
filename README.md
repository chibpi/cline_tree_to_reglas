# Decision Tree Rule Extraction System

## Description

This project provides a comprehensive machine learning system for text classification using Decision Trees and rule extraction. The system trains Decision Tree classifiers on text data, extracts interpretable logical rules from the trained models, and provides tools for comparing classifier performance.

The primary goal is to bridge the gap between complex machine learning models and human-interpretable decision rules, making AI classification more transparent and explainable. The project is particularly focused on Spanish text classification for procedural document categorization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Required Dependencies

Install the required Python packages:

```bash
pip install scikit-learn pandas numpy matplotlib joblib
```

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd cline_tree_to_reglas
```

2. Ensure you have the required data files:
   - `textos_procesados.csv` - Preprocessed text data
   - `spanish_stopwords.txt` - Spanish stopwords list

## Usage

### Training a Decision Tree Classifier

```bash
python dectree_clasificacion.py
```

This script:
- Loads and preprocesses text data
- Trains a Decision Tree classifier with TF-IDF vectorization
- Generates visualizations of the decision tree
- Saves the trained model, vectorizer, and categories

### Extracting Rules from Decision Tree

```bash
python extract_rules.py
```

This script:
- Loads the trained Decision Tree model
- Extracts logical rules from the tree structure
- Simplifies rules for sequential application
- Saves rules to `decision_tree_rules_cvo.txt`

### Applying Rules for Classification

```bash
python apply_rules.py
```

This script applies the extracted rules to classify new text data using the interpretable rule-based approach.

### Comparing Classifier Performance

```bash
python compare_classifiers.py
```

This script compares the performance of the original Decision Tree classifier with the rules-based classifier and generates a comprehensive comparison report.

### Finding Specific Control Rules

```bash
python find_pnc_control_rules.py
```

This script extracts specific rules related to PNC control procedures.

## Features

### Core Functionality

- **Text Classification**: Multi-class classification of text documents using Decision Trees
- **Rule Extraction**: Automatic extraction of interpretable logical rules from trained Decision Trees
- **Model Persistence**: Save and load trained models, vectorizers, and category mappings
- **Performance Comparison**: Compare Decision Tree vs. Rules-based classifier performance
- **Visualization**: Generate PDF visualizations of decision trees

### Data Processing

- **TF-IDF Vectorization**: Convert text to numerical features with n-gram support (1-5 grams)
- **Spanish Language Support**: Built-in Spanish stopwords filtering
- **Text Preprocessing**: Automatic handling of missing values and text normalization

### Rule-Based Classification

- **Sequential Rule Application**: Rules are applied in order, with early stopping on match
- **Confidence Scoring**: Each rule includes confidence metrics
- **Human-Readable Rules**: Logical conditions expressed in natural language format
- **Rule Simplification**: Complex decision tree paths converted to simplified logical conditions

### Model Evaluation

- **Cross-Validation**: Automated model evaluation with cross-validation
- **Performance Metrics**: Accuracy, precision, recall, and F1-score reporting
- **Confusion Matrix Analysis**: Detailed class-wise performance analysis
- **Comparative Reports**: Side-by-side comparison of different classification approaches

## Configuration

### Model Parameters

The Decision Tree classifier can be configured with the following parameters in `dectree_clasificacion.py`:

```python
# TF-IDF Vectorizer parameters
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1,5),  # 1-5 grams
    max_features=1000   # Top 1000 features
)

# Decision Tree parameters
clf = DecisionTreeClassifier(
    random_state=42,
    min_samples_leaf=2,
    ccp_alpha=optimal_alpha  # Automatically optimized
)
```

### Data Configuration

- **Input Data**: CSV files with text data and labels
- **Text Column**: "asunto preprocesado" (preprocessed subject)
- **Label Column**: "procedimiento" (procedure/category)
- **Stopwords**: Custom Spanish stopwords list

#### Dataset Structure

The main dataset file `textos_procesados.csv` contains the following columns:

- **ruta**: File path or document location
- **procedimiento**: Procedure/category label (target variable for classification)
- **tramite**: Administrative process type
- **asunto**: Original subject text
- **expone_solicita**: Exposition and request section
- **asunto_expone_solicita**: Combined subject, exposition and request text
- **asunto preprocesado**: Preprocessed subject text (used for model training)

### Rule Extraction Settings

Rule extraction parameters can be adjusted in `extract_rules.py`:

```python
# Confidence threshold for including rules
confidence_threshold = 0.5  # Only include rules with >50% confidence

# Rule simplification settings
# Converts complex conditions to presence/absence of n-grams
```

### Output Files

The system generates several output files:

- **Models**: `clasificador_*.pkl` (trained classifiers)
- **Vectorizers**: `vectorizer_*.pkl` (TF-IDF vectorizers)
- **Categories**: `categorias_*.pkl` (class labels)
- **Rules**: `decision_tree_rules_*.txt` (extracted logical rules)
- **Visualizations**: `decision_tree*.pdf` (tree diagrams)
- **Evaluation**: CSV files with performance metrics and predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Project Structure

```
cline_tree_to_reglas/
├── dectree_clasificacion.py      # Main training script
├── extract_rules.py              # Rule extraction script
├── apply_rules.py                # Rule application script
├── compare_classifiers.py        # Performance comparison
├── find_pnc_control_rules.py     # Specific rule extraction
├── textos_procesados.csv         # Main dataset
├── spanish_stopwords.txt         # Spanish stopwords
├── clasificador_*.pkl           # Trained models
├── vectorizer_*.pkl             # TF-IDF vectorizers
├── categorias_*.pkl             # Category mappings
├── decision_tree_rules_*.txt    # Extracted rules
├── decision_tree*.pdf           # Tree visualizations
└── evaluation_*.csv             # Performance results
```

## Performance Summary

Based on current evaluations:
- **Decision Tree Accuracy**: 61.74%
- **Rules-Based Accuracy**: 30.55%
- **Number of Classes**: 17
- **Dataset Size**: 622 samples

The Decision Tree classifier significantly outperforms the rules-based approach in accuracy, but the rules-based system provides interpretable, human-readable decision logic.
