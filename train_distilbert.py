#! /usr/bin/env python3

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('labeled_messages.csv')
    
    # Drop rows with missing messages
    df = df.dropna(subset=['Message'])
    
    # Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, val_df

# Tokenize the data
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['Message'], padding='max_length', truncation=True)

# Convert dataframes to datasets
def create_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset

# Main function to run the training and evaluation
def main():
    # Load and preprocess data
    train_df, val_df = load_and_preprocess_data()
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Move model to GPU if available
    model.to(device)
    
    # Create datasets
    train_dataset = create_dataset(train_df, tokenizer)
    val_dataset = create_dataset(val_df, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        output_dir='./results',
        num_train_epochs=3,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    results = trainer.evaluate(val_dataset)
    print(results)
    
    # Save the model and tokenizer
    model.save_pretrained('./distilbert-finetuned')
    tokenizer.save_pretrained('./distilbert-finetuned')

if __name__ == "__main__":
    main()

