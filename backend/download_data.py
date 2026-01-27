"""
Download and prepare the AsyLex dataset for training.
"""
from datasets import load_dataset
import pandas as pd
import os

def main():
    print("Downloading AsyLex outcome classification dataset...")
    
    # Load the outcome classification split (has train/test with labels)
    dataset = load_dataset("clairebarale/AsyLex", "outcome_classification")
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save as CSV for easier inspection
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print("\nDataset columns:", train_df.columns.tolist())
    print("\nOutcome distribution (train):")
    print(train_df['decision_outcome'].value_counts())
    print("\n0 = Rejected, 1 = Granted, 2 = Uncertain")
    
    # Also load the entities dataset for richer features
    print("\nDownloading entity extraction data...")
    entities = load_dataset("clairebarale/AsyLex", "casecover_entities_outcome")
    entities_df = pd.DataFrame(entities['train'])
    entities_df.to_csv("data/entities.csv", index=False)
    print(f"Entities dataset: {len(entities_df)} rows")
    print("Entity columns:", entities_df.columns.tolist())
    
    print("\nData downloaded successfully to ./data/")

if __name__ == "__main__":
    main()
