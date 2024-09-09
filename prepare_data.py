#! /usr/bin/env python3

import pandas as pd

# Load chat log and kicked messages
chatlog_df = pd.read_csv('chatlog.csv')
kicked_messages_df = pd.read_csv('kicked_messages.csv')

# Remove NaN messages from both datasets
chatlog_df = chatlog_df.dropna(subset=['Message'])
kicked_messages_df = kicked_messages_df.dropna(subset=['Message'])

# Add labels to the kicked messages (label = 1)
kicked_messages_df['label'] = 1

# Filter non-kicked messages and use .loc to avoid the SettingWithCopyWarning
non_kicked_df = chatlog_df[~chatlog_df['Message'].isin(kicked_messages_df['Message'])].copy()
non_kicked_df.loc[:, 'label'] = 0  # Use .loc[] for setting the 'label' column

# Combine datasets
combined_df = pd.concat([kicked_messages_df, non_kicked_df])

# Save to a new CSV file without NaN messages
combined_df.to_csv('labeled_messages.csv', index=False)

