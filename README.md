# DNA Binding Protein Classification

## Introduction

This project focuses on DNA Binding Protein classification using various machine learning models. Our goal is to explore different feature extraction methods and classifiers to accurately distinguish between binding and non-binding
proteins. By both biological knowledge (amino acid distribution, term frequency
inverse document term, physicochemical properties) and natural language processing techniques (embedding, integer encoding) with both traditional MLs
(Logistic Regression, Naive Bayes, KNN, Support Vector Machine, Decision
Tree, Random Forest) and Deep Learning models (CNN, ProtCNN, Large Language Model), we aim to identify the most effective approach for protein classification.


## Project Structure

### Code Folder

The notebooks should be accessed in the following order:

1. `Exploratory Data Analysis.ipynb`
2. `AAD.ipynb` - Amino Acid Distribution and TF-IDF
3. `TF-IDF.ipynb` - Term Frequency-Inverse Document Frequency
4. `Physicochemical Properties.ipynb`
5. `pseAAC.ipynb` - Pseudo Amino Acid Composition
6. `CNN.ipynb` - Convolutional Neural Network
7. `ProtBert.ipynb` - Large Language Model
8. `Comparison Plots between Models`

## Dataset and Files

The dataset used in this project is titled **DNA Binding Protein**.

### Dataset Files

The `Model & DataSet` folder contains the following datasets:

| File | Description |
|---|---|
| `Train.fasta` | Original training dataset |
| `Test.fasta` | Original testing dataset |
| `Train_valid.fasta` | Cleaned training dataset containing only sequences with common amino acids |
| `Test_valid.fasta` | Cleaned testing dataset containing only sequences with common amino acids |
| `DNA_Test.csv` | Cleaned testing dataset containing physicochemical properties |
| `DNA_Train.csv` | Cleaned training dataset containing physicochemical properties |

Make sure the training and testing files are located in the same root directory as the notebooks and model files.

---

## Saved Models

### Amino Acid Distribution Models

| File | Description |
|---|---|
| `LR_Amino_Acid_Distribution_Full.joblib` | Logistic Regression with full 20 amino acid distribution features |
| `LR_Amino_Acid_Distribution_Selected.joblib` | Logistic Regression with selected amino acid distribution features |
| `NB_Amino_Acid_Distribution_Full.joblib` | Naive Bayes with full 20 amino acid distribution features |
| `NB_Amino_Acid_Distribution_Selected.joblib` | Naive Bayes with selected amino acid distribution features |
| `KNN_Amino_Acid_Distribution_Full.joblib` | K-Nearest Neighbours with full 20 amino acid distribution features |
| `KNN_Amino_Acid_Distribution_Selected.joblib` | K-Nearest Neighbours with selected amino acid distribution features |
| `DT_Amino_Acid_Distribution_Full.joblib` | Decision Tree with full 20 amino acid distribution features |
| `DT_Amino_Acid_Distribution_Selected.joblib` | Decision Tree with selected amino acid distribution features |
| `RF_Amino_Acid_Distribution_Full.joblib` | Random Forest with full 20 amino acid distribution features |
| `RF_Amino_Acid_Distribution_Selected.joblib` | Random Forest with selected amino acid distribution features |
| `SVM_Amino_Acid_Distribution_Full.joblib` | Support Vector Machine with full 20 amino acid distribution features |
| `SVM_Amino_Acid_Distribution_Selected.joblib` | Support Vector Machine with selected amino acid distribution features |

### Physicochemical Property Models

| File | Description |
|---|---|
| `LR_Physicochemical_Properties.joblib` | Logistic Regression using physicochemical properties |
| `NB_Physicochemical_Properties.joblib` | Naive Bayes using physicochemical properties |
| `KNN_Physicochemical_Properties.joblib` | K-Nearest Neighbours using physicochemical properties |
| `DT_Physicochemical_Properties.joblib` | Decision Tree using physicochemical properties |
| `RF_Physicochemical_Properties.joblib` | Random Forest using physicochemical properties |
| `SVM_Physicochemical_Properties.joblib` | Support Vector Machine using physicochemical properties |

### pseAAC Models

| File | Description |
|---|---|
| `LR_pseAAC.joblib` | Logistic Regression using pseudo amino acid composition |
| `NB_pseAAC.joblib` | Naive Bayes using pseudo amino acid composition |
| `KNN_pseAAC.joblib` | K-Nearest Neighbours using pseudo amino acid composition |
| `DT_pseAAC.joblib` | Decision Tree using pseudo amino acid composition |
| `RF_pseAAC.joblib` | Random Forest using pseudo amino acid composition |
| `SVM_pseAAC.joblib` | Support Vector Machine using pseudo amino acid composition |

### Deep Learning Models

| File or Folder | Description |
|---|---|
| `CNN1.h5` | CNN model using embedding |
| `CNN2.h5` | CNN model using amino acid integer encoding |
| `ProtCNN1.h5` | ProtCNN model using embedding |
| `ProtCNN2.h5` | ProtCNN model using amino acid integer encoding |
| `trained_model/` | Trained model weights and architecture configuration |
| `trained_tokenizer/` | Tokenizer files required for preprocessing input data |

The `trained_model/` folder contains files such as:

- `pytorch_model.bin`
- `config.json`

The `trained_tokenizer/` folder contains files such as:

- `vocab.txt`
- `tokenizer_config.json`

---

## Installation

Make sure `requirements.txt` is located in the root directory.

Install the required libraries with:

```bash
pip install -r requirements.txt
```

The required libraries include:

```txt
biopython
imbalanced-learn
propy3
scikit-learn
tensorflow
tqdm
pandas
numpy
keras
seaborn
kat
statsmodels
```

---

## Notebook Descriptions

## 1. Exploratory Data Analysis

This notebook is used to understand the dataset before model training.

It includes:

- Reading and processing protein sequence data
- Exploring class 0 and class 1 amino acid composition
- Analyzing amino acid distributions
- Analyzing protein sequence length distributions
- Plotting graphs for data exploration

### Running the Notebook

The notebook can be run in a Jupyter Notebook environment.

If using Google Colab, make sure the training and testing data are uploaded and that the file paths are adjusted correctly.

---

## 2. Amino Acid Frequency Distribution and TF-IDF

This notebook uses biological features such as amino acid frequency distribution and TF-IDF to classify proteins using traditional machine learning models.

### Running the Notebook

The notebook can be run in Jupyter Notebook.

If using Google Colab, make sure the training and testing data are uploaded.

### Saving and Loading Models

To load a saved model:

```python
import joblib

model = joblib.load(filename)
```

---

## 3. Physicochemical Properties

This notebook explores how physicochemical properties of amino acids, such as hydrophobicity and net charge, can be used for protein classification with traditional machine learning models.

### Running the Notebook

The notebook can be run in Jupyter Notebook.

If using Google Colab, make sure the training and testing data are uploaded.

### Saving and Loading Models

To load a saved model:

```python
import joblib

model = joblib.load(filename)
```

---

## 4. pseAAC

This notebook explores pseudo amino acid composition as a feature extraction method.

pseAAC improves standard amino acid distribution by incorporating additional sequence pattern information and physicochemical properties.

### Running the Notebook

The notebook can be run in Jupyter Notebook.

If using Google Colab, make sure the training and testing data are uploaded.

> Warning: this notebook may take around 22 hours to run.

### Saving and Loading Models

To load a saved model:

```python
import joblib

model = joblib.load(filename)
```

---

## 5. CNN

This notebook uses Convolutional Neural Networks for protein classification with NLP-based techniques:

- Embedding
- Amino acid integer encoding

This notebook uses:

- `Train_valid.fasta`
- `Test_valid.fasta`

### Running the Notebook

The notebook can be run in Jupyter Notebook.

If using Google Colab, make sure the training and testing data are uploaded.

Using a GPU is recommended for faster training.

> Warning: running on CPU may take around 24 hours, while running on GPU may take around 2 hours.

### Saving and Loading Models

The model can be saved and loaded using an H5 file.

An H5 file includes:

- Model weights
- Model architecture
- Model compilation details
- Model optimizer state

To load a saved model:

```python
from keras.models import load_model

model = load_model(file_path)
```

---

## 6. LLM / ProtBERT

This notebook uses a Large Language Model for protein classification.

The model used is **ProtBERT**, a pretrained protein language model.

### Running the Notebook

The notebook can be run in Jupyter Notebook.

If using Google Colab, make sure the training and testing data are uploaded.

> Warning: training may take around 2.5 hours.

### Loading the Model

Example loading syntax:

```python
from transformers import BertModel

model = BertModel.from_pretrained(filename)
```

---

## Notes

- Ensure all datasets are placed in the correct root directory before running the notebooks.
- Use the cleaned `Train_valid.fasta` and `Test_valid.fasta` files for CNN-based models.
- GPU usage is recommended for deep learning notebooks.
- Some notebooks may take several hours to complete.
- Public URLs are not included in this README.
