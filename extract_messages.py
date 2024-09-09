#! /usr/bin/env python3

import csv

# Load the kicked users into a set for fast lookups
with open("kicked_users.txt", "r") as f:
    kicked_users = set(line.strip() for line in f)

# Define the header manually
header = ["Timestamp", "From", "To", "Message", "isPM", "isDeleted", "Channel"]

# Open the chatlog.csv and filter messages by kicked users
with open("chatlog.csv", "r") as infile, open("kicked_messages.csv", "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write the header row to the output CSV
    writer.writerow(header)
    
    # Iterate through chatlog.csv and filter rows where the sender is in kicked_users
    for row in reader:
        sender = row[1]  # Assuming the sender is in the second column ('From')
        if sender in kicked_users:
            writer.writerow(row)

print("Filtered kicked messages written to kicked_messages.csv")

