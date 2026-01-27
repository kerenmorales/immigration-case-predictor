"""
Train a transformer-based case outcome prediction model.
Uses DistilBERT for efficiency (good balance of speed and accuracy).
"""
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

MODEL_NAME = "distilbert-base-uncased"  # Fast and effective
OUTPUT_DIR = "models/transformer"
NUM_LABELS = 3  # Rejected, Granted, Uncertain

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Loading AsyLex dataset...")
    dataset = load_dataset("clairebarale/AsyLex", "outcome_classification")
    
    # Check columns and find text column
    print("Columns:", dataset['train'].column_names)
    
    # Determine text column
    text_col = None
    for col in ['text', 'case_text', 'content', 'sentence']:
        if col in dataset['train'].column_names:
            text_col = col
            break
    
    if text_col is None:
        # Use first string column that's not the label
        for col in dataset['train'].column_names:
            if col != 'decision_outcome':
                text_col = col
                break
    
    print(f"Using '{text_col}' as text column")
    
    def preprocess(examples):
        """Tokenize the text."""
        texts = [str(t) if t else "" for t in examples[text_col]]
        # Truncate to 512 tokens (DistilBERT limit)
        return tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=False  # Dynamic padding in collator
        )
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=[c for c in dataset['train'].column_names if c != 'decision_outcome']
    )
    
    # Rename label column
    tokenized = tokenized.rename_column('decision_outcome', 'labels')
    
    print(f"Train size: {len(tokenized['train'])}")
    print(f"Test size: {len(tokenized['test'])}")
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "Rejected", 1: "Granted", 2: "Uncertain"},
        label2id={"Rejected": 0, "Granted": 1, "Uncertain": 2}
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training...")
    print("This may take 1-2 hours on CPU, ~15-30 min on GPU")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    results = trainer.evaluate()
    print(f"Results: {results}")
    
    # Save the best model
    print(f"\nSaving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    print("\nTraining complete!")
    print(f"Model saved to {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
