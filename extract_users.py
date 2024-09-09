#! /usr/bin/env python3

import csv

# Input and output file names
chatlog_file = 'chatlog.csv'
kicked_users_file = 'kicked_users.txt'

# Initialize an empty set to store kicked users (avoid duplicates)
kicked_users = set()

# Extract kicked users from system messages
with open(chatlog_file, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        message = row['Message']
        sender = row['From']
        if sender == "System" and "has been kicked" in message:
            username = message.split(" ")[0]
            kicked_users.add(username)

# Write the kicked users to a file
with open(kicked_users_file, 'w', encoding='utf-8') as file:
    for user in kicked_users:
        file.write(user + '\n')

print(f"Extracted {len(kicked_users)} kicked users to {kicked_users_file}")
