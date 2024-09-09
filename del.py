#! /usr/bin/env python3

import pandas as pd
from transformers import DistilBertTokenizer

# Load the pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load your chatlog CSV
chatlog_df = pd.read_csv('chatlog.csv')

# Initialize counters and lists
over_limit_count = 0
over_limit_messages = []
nan_messages = []

# Loop through each message in the CSV and check its token length
for index, row in chatlog_df.iterrows():
    message = row['Message']  # Adjust the column name if needed

    # Check for NaN messages
    if pd.isna(message):
        nan_messages.append(row['Timestamp'])
        continue

    # Tokenize the message
    tokens = tokenizer.encode(str(message), truncation=False)  # Ensure message is treated as a string

    # Check if token length exceeds 512
    if len(tokens) > 512:
        over_limit_count += 1
        over_limit_messages.append((row['Timestamp'], len(tokens)))

# Output the result
print(f"Number of messages exceeding 512 tokens: {over_limit_count}")
if over_limit_messages:
    print("Messages exceeding the limit (Timestamp, Token Count):")
    for msg in over_limit_messages:
        print(msg)

# Output NaN messages
print(f"Number of NaN messages: {len(nan_messages)}")
if nan_messages:
    print("NaN messages found at these timestamps:")
    for timestamp in nan_messages:
        print(timestamp)

