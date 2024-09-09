#! /usr/bin/env python3

# Load usernames from kicked_users.txt
with open("kicked_users.txt", "r") as f:
    kicked_users = set(line.strip() for line in f)

# Load usernames from members.txt
with open("members.txt", "r") as f:
    members = set(line.strip() for line in f)

# Find any overlaps between kicked_users and members
overlaps = kicked_users.intersection(members)

# Output overlapping users
if overlaps:
    print("The following users are in both kicked_users.txt and members.txt:")
    for user in overlaps:
        print(user)
else:
    print("No overlapping users found between kicked_users.txt and members.txt.")

# Remove overlapping users from kicked_users
updated_kicked_users = kicked_users - overlaps

# Write the updated kicked_users.txt
with open("kicked_users.txt", "w") as f:
    for user in sorted(updated_kicked_users):
        f.write(user + "\n")

print("Updated kicked_users.txt")

