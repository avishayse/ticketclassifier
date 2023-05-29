import csv

# Define the ticket descriptions
ticket_descriptions = [
    "Cannot login to my account",
    "Payment not going through",
    "Product not working after update",
    "Need assistance with installation",
    "Forgot password",
    "Order status inquiry",
    "Billing discrepancy",
    "Feature request",
    "Network connectivity issues",
    "Email not receiving messages"
]

# Define the ticket categories
ticket_categories = [
    "Login Issues",
    "Payment Problems",
    "Technical Issues",
    "Installation Help",
    "Login Issues",
    "Order Inquiry",
    "Billing Issues",
    "Feature Request",
    "Technical Issues",
    "Email Problems"
]

# Create a CSV file for the tickets
csv_file_path = "support_tickets.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Ticket Description", "Category"])

    for description, category in zip(ticket_descriptions, ticket_categories):
        writer.writerow([description, category])

print(f"CSV file '{csv_file_path}' generated successfully.")

