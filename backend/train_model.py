"""
Train the case outcome prediction model.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

OUTCOME_LABELS = {0: "Rejected", 1: "Granted", 2: "Uncertain"}

def load_data():
    """Load the training and test data."""
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    return train_df, test_df

def create_model():
    """Create the ML pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0
        ))
    ])

def train():
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Check what text column we have
    print("Columns:", train_df.columns.tolist())
    
    # The dataset should have case text and decision_outcome
    # Find the text column (might be 'text', 'case_text', or similar)
    text_col = None
    for col in ['text', 'case_text', 'content', 'document']:
        if col in train_df.columns:
            text_col = col
            break
    
    if text_col is None:
        # If no obvious text column, use all string columns concatenated
        string_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        if 'decision_outcome' in string_cols:
            string_cols.remove('decision_outcome')
        if string_cols:
            text_col = string_cols[0]
            print(f"Using column '{text_col}' as text input")
        else:
            raise ValueError("No text column found in dataset")
    
    # Prepare data
    X_train = train_df[text_col].fillna("").astype(str)
    y_train = train_df['decision_outcome']
    X_test = test_df[text_col].fillna("").astype(str)
    y_test = test_df['decision_outcome']
    
    # Filter out uncertain cases for cleaner binary classification (optional)
    # For now, keep all 3 classes
    
    print(f"\nTraining on {len(X_train)} samples...")
    print(f"Class distribution:\n{y_train.value_counts()}")
    
    # Train model
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(OUTCOME_LABELS.values())))
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/outcome_predictor.joblib")
    print("\nModel saved to models/outcome_predictor.joblib")
    
    # Save feature importance (top words for each class)
    save_feature_importance(model)

def save_feature_importance(model):
    """Extract and save the most important features for each outcome."""
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    
    feature_names = vectorizer.get_feature_names_out()
    
    importance = {}
    for i, label in OUTCOME_LABELS.items():
        if len(classifier.classes_) > 2:
            coef = classifier.coef_[i]
        else:
            coef = classifier.coef_[0] if i == 1 else -classifier.coef_[0]
        
        top_indices = np.argsort(coef)[-20:][::-1]
        top_features = [(feature_names[j], coef[j]) for j in top_indices]
        importance[label] = top_features
    
    # Save as readable text
    with open("models/feature_importance.txt", "w") as f:
        for label, features in importance.items():
            f.write(f"\n=== Top predictors for {label} ===\n")
            for word, score in features:
                f.write(f"  {word}: {score:.4f}\n")
    
    print("Feature importance saved to models/feature_importance.txt")

if __name__ == "__main__":
    train()
