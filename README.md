# Fine-Tuning Transformer Models for Cross-Domain Text Classification

A comprehensive Natural Language Processing project demonstrating transfer learning and fine-tuning of transformer models for sentiment analysis. This project fine-tunes DistilBERT on the Sentiment140 Twitter dataset, visualizes attention mechanisms, and performs hyperparameter optimization.

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Components](#-project-components)
- [Technologies Used](#️-technologies-used)
- [Dataset Information](#-dataset-information)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Attention Visualization](#-attention-visualization)
- [Evaluation Metrics](#-evaluation-metrics)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Results](#-results)
- [Learning Outcomes](#-learning-outcomes)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)
- [References](#-references)

## Overview

This project implements a complete pipeline for fine-tuning transformer models on text classification tasks. Using the Sentiment140 Twitter dataset, we fine-tune DistilBERT (a lightweight BERT variant) for binary sentiment classification. The project covers data preprocessing, model training, attention visualization, evaluation, and hyperparameter optimization.

**Key Highlights:**
- Fine-tunes DistilBERT for sentiment analysis
- Processes 12,000 tweets with binary classification (positive/negative)
- Achieves 81.82% accuracy on test set
- Visualizes multi-head attention across 6 transformer layers
- Performs grid search for optimal hyperparameters

##  Key Features

- **Transfer Learning**: Leverages pre-trained DistilBERT model
- **Sentiment Classification**: Binary classification (positive/negative) on Twitter data
- **Attention Visualization**: Multi-layer, multi-head attention heatmaps
- **Hyperparameter Tuning**: Grid search over learning rates, batch sizes, and epochs
- **Comprehensive Evaluation**: Accuracy, F1-score, precision, recall, and confusion matrix
- **Real-World Data**: Uses noisy social media text with slang, emojis, and abbreviations

## Project Components

### Step 1: Dataset Selection and Domain Understanding
- Load Sentiment140 dataset from Hugging Face
- Filter and preprocess 12,000 tweets
- Convert sentiment labels (0=negative, 4=positive) to binary (0/1)
- Split into train/validation/test sets (10k/1k/1k)

### Step 2: Transfer Learning and Fine-Tuning
- Load pre-trained DistilBERT model with classification head
- Tokenize text using DistilBERT tokenizer
- Configure training arguments (learning rate, batch size, epochs)
- Train model using Hugging Face Trainer API
- Save fine-tuned model for inference

### Step 3: Attention Visualization and Interpretation
- Extract attention weights from all 6 transformer layers
- Visualize attention patterns for each attention head
- Analyze how different layers focus on different tokens
- Interpret attention patterns across model depth

### Step 4: Model Evaluation & Hyperparameter Tuning
- Evaluate model on test set with classification metrics
- Generate confusion matrix and classification report
- Perform grid search over hyperparameters
- Compare performance across different configurations

## Technologies Used

- **Python 3.x**: Core programming language
- **Transformers (Hugging Face)**: Pre-trained models and training utilities
- **DistilBERT**: Lightweight BERT model for sequence classification
- **PyTorch**: Deep learning framework
- **Hugging Face Datasets**: Dataset loading and preprocessing
- **scikit-learn**: Evaluation metrics (accuracy, F1-score, confusion matrix)
- **Matplotlib & Seaborn**: Data visualization and attention heatmaps
- **NumPy**: Numerical computations

## Dataset Information

### Sentiment140 Dataset

- **Source**: Kaggle - Sentiment140 by Kazanova (via Hugging Face)
- **Total Size**: 160,000 tweets (project uses 12,000 subset)
- **Task**: Binary sentiment classification
- **Labels**: 
  - `0` = Negative sentiment
  - `4` = Positive sentiment (converted to `1` for binary classification)
- **Features**:
  - `text`: Tweet content
  - `sentiment`: Original label (0 or 4)
  - `label`: Binary label (0 or 1)
  - `date`: Tweet timestamp
  - `user`: Twitter username

### Dataset Split

- **Training Set**: 9,999 samples
- **Validation Set**: 1,000 samples
- **Test Set**: 1,001 samples

### Data Characteristics

- Real-world noisy data (slang, emojis, abbreviations)
- Informal language typical of social media
- Variable-length text sequences
- Preprocessed and annotated for sentiment

## Installation

### Prerequisites

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn numpy
```

### Install Specific Versions

```bash
pip install transformers>=4.51.3
pip install datasets>=3.5.1
pip install torch
```

## Quick Start

```python
# 1. Install dependencies
!pip install transformers datasets torch scikit-learn matplotlib seaborn

# 2. Load dataset
from datasets import load_dataset
dataset = load_dataset("sentiment140", trust_remote_code=True)

# 3. Filter and prepare data
dataset = dataset.filter(lambda x: x['sentiment'] in [0, 4])
dataset = dataset.map(lambda x: {"label": 0 if x['sentiment'] == 0 else 1})
small_dataset = dataset["train"].shuffle(seed=42).select(range(12000))

# 4. Load model and tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# 5. Tokenize data
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = small_dataset.map(tokenize_function, batched=True)

# 6. Train model
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

trainer.train()
```

## Usage

### Loading and Preprocessing Data

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load Sentiment140 dataset
dataset = load_dataset("sentiment140", trust_remote_code=True)

# Filter for binary classification
dataset = dataset.filter(lambda x: x['sentiment'] in [0, 4])
dataset = dataset.map(lambda x: {"label": 0 if x['sentiment'] == 0 else 1})

# Select subset
small_dataset = dataset["train"].shuffle(seed=42).select(range(12000))

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = small_dataset.map(tokenize_function, batched=True, remove_columns=["sentiment", "date", "query", "user", "text"])
```

### Fine-Tuning the Model

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["validation"],
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
trainer.save_model("./fine_tuned_model")
```

### Evaluating the Model

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on test set
predictions = trainer.predict(final_dataset["test"])
pred_labels = predictions.predictions.argmax(axis=-1)
true_labels = predictions.label_ids

# Classification report
print(classification_report(true_labels, pred_labels, digits=4))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

### Attention Visualization

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load fine-tuned model with attention
model = DistilBertForSequenceClassification.from_pretrained(
    "./fine_tuned_model", 
    output_attentions=True
)
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")

# Example sentence
sentence = "I love this movie so much"
inputs = tokenizer(sentence, return_tensors="pt")

# Get attention weights
model.eval()
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

# Get tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Visualize attention for each layer and head
num_layers = len(attentions)
fig, axes = plt.subplots(2, num_layers, figsize=(3 * num_layers, 6))

for layer_idx, attention in enumerate(attentions):
    # Head 1
    ax0 = axes[0, layer_idx]
    attn0 = attention[0, 0]  # batch=0, head=0
    sns.heatmap(attn0.numpy(), xticklabels=tokens, yticklabels=tokens,
                ax=ax0, cmap="viridis", cbar=False)
    ax0.set_title(f"Layer {layer_idx+1} – Head 1", fontsize=8)
    
    # Head 2
    ax1 = axes[1, layer_idx]
    attn1 = attention[0, 1]  # batch=0, head=1
    sns.heatmap(attn1.numpy(), xticklabels=tokens, yticklabels=tokens,
                ax=ax1, cmap="viridis", cbar=False)
    ax1.set_title(f"Layer {layer_idx+1} – Head 2", fontsize=8)

plt.tight_layout()
plt.show()
```

### Hyperparameter Tuning

```python
from transformers import Trainer, TrainingArguments
import itertools

# Define parameter grid
param_grid = {
    "learning_rate": [5e-6, 2e-5],
    "per_device_train_batch_size": [8, 16],
    "num_train_epochs": [2, 3],
}

best_score = 0.0
best_params = None

# Grid search
for lr, bs, epochs in itertools.product(
    param_grid["learning_rate"],
    param_grid["per_device_train_batch_size"],
    param_grid["num_train_epochs"],
):
    args = TrainingArguments(
        output_dir=f"./results_lr{lr}_bs{bs}_ep{epochs}",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=final_dataset["train"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Evaluate and track best
    pred = trainer.predict(final_dataset["validation"])
    # ... evaluate and update best_score/best_params
```

##  Model Architecture

### DistilBERT

- **Base Model**: `distilbert-base-uncased`
- **Architecture**: 6 transformer layers (vs. 12 in BERT)
- **Attention Heads**: 12 heads per layer (2 heads per layer in DistilBERT)
- **Hidden Size**: 768 dimensions
- **Parameters**: ~66 million (vs. ~110 million in BERT)
- **Classification Head**: Linear layer on top of [CLS] token

### Model Configuration

- **Input**: Tokenized text sequences (max length: 128 tokens)
- **Output**: Binary classification (2 classes: negative/positive)
- **Special Tokens**: [CLS], [SEP], [PAD]
- **Tokenization**: WordPiece tokenization

##  Attention Visualization

### Attention Patterns Observed

**Early Layers (1-2):**
- Broad attention distribution
- Strong diagonal patterns (self-attention)
- [CLS] token attends to all tokens for global context

**Middle Layers (3-4):**
- Emerging contextual relationships
- Selective focus on relevant tokens
- Off-diagonal attention patterns appear

**Late Layers (5-6):**
- Structured patterns (blocky/cross-shaped)
- Focused attention on key tokens
- Syntactic and semantic groupings
- Task-specific attention alignment

### Multi-Head Attention

- **Head 1**: Often captures global context and sequence-level information
- **Head 2**: Shows different patterns, focusing on specific aspects
- Demonstrates the value of multi-head attention for diverse representations

##  Evaluation Metrics

### Test Set Performance

- **Accuracy**: 81.82%
- **F1-Score (Macro Avg)**: 81.82%
- **F1-Score (Weighted Avg)**: 81.82%

### Per-Class Metrics

**Class 0 (Negative):**
- Precision: 82.97%
- Recall: 81.02%
- F1-Score: 81.98%

**Class 1 (Positive):**
- Precision: 80.68%
- Recall: 82.65%
- F1-Score: 81.65%

### Confusion Matrix

```
                Predicted
              Negative  Positive
Actual Negative   414      97
       Positive    85      405
```

**Interpretation:**
- Well-balanced performance across both classes
- Slight bias: Class 0 has higher precision, Class 1 has higher recall
- Model is not heavily biased toward either class

## 🔧 Hyperparameter Tuning

### Grid Search Parameters

- **Learning Rates**: `[5e-6, 2e-5]`
- **Batch Sizes**: `[8, 16]`
- **Epochs**: `[2, 3]`
- **Total Combinations**: 8 runs (2 × 2 × 2)

### Observations

- Lower learning rates (5e-6) stabilize training
- More epochs generally improve performance
- Smaller batch sizes can lead to better convergence
- Training loss decreases consistently across configurations

##  Learning Outcomes

After completing this project, I understand:

- How to fine-tune pre-trained transformer models for downstream tasks
- Transfer learning concepts and their practical applications
- Attention mechanisms and how to visualize them
- Hyperparameter tuning strategies for transformer models
- Evaluation metrics for classification tasks
- Working with real-world noisy text data
- Using Hugging Face Transformers library effectively

##  Project Structure

```
Fine-Tuning-Transformer-Models-for-Cross-Domain-Text-Classification/
├── Fine-Tuning-Transformer-Models-for-Cross-Domain-Text-Classification.ipynb
└── README.md                             # This file
```

## Future Enhancements

- [ ] Experiment with other transformer models (BERT, RoBERTa, ELECTRA)
- [ ] Implement cross-domain evaluation on different datasets
- [ ] Add more sophisticated hyperparameter tuning (Optuna, Ray Tune)
- [ ] Implement early stopping and learning rate scheduling
- [ ] Add data augmentation techniques
- [ ] Implement ensemble methods
- [ ] Deploy model as a web service
- [ ] Add support for multi-class classification
- [ ] Implement interpretability tools (LIME, SHAP)
- [ ] Add support for other languages

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Sentiment140 Dataset](https://huggingface.co/datasets/sentiment140)
- [Attention Is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)

## 👤 Author

**Khanh Le**

## License

This project is for educational purposes. The Sentiment140 dataset is available under its original license. DistilBERT model is available under Apache 2.0 license.



