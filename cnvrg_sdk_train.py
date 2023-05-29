import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from cnvrgv2 import Experiment
from cnvrgv2 import Cnvrg

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a text classification model')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training')
args = parser.parse_args()

# Load the dataset
dataset_path = "/cnvrg/support_tickets.csv"
df = pd.read_csv(dataset_path)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Ticket Description'].values, df['Category'].values, test_size=0.2, random_state=42
)

# Label encoding for the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Filter out unseen labels in the validation set
valid_labels_filtered = [label for label in val_labels if label in label_encoder.classes_]
val_labels_encoded = label_encoder.transform(valid_labels_filtered)

# Load the pre-trained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenize the texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Create PyTorch datasets
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextClassificationDataset(train_encodings, train_labels_encoded)
val_dataset = TextClassificationDataset(val_encodings, val_labels_encoded)

# Define training parameters
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

e = Experiment()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_correct += torch.sum(preds == labels)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_dataset)
    val_accuracy = val_accuracy.item()
                                     
    # Log metrics using cnvrg
    e.log_param("epoch", epoch+1)
    e.log_param("train_loss", train_loss)
    e.log_param("val_loss", val_loss)
    e.log_param("val_accuracy", val_accuracy)

    # Print metrics to standard output
    print(f"cnvrg_linechart_epoch value: {epoch+1}")
    print(f"cnvrg_linechart_train_loss value: {train_loss:.4f}")
    print(f"cnvrg_linechart_val_loss value: {val_loss:.4f}")
    print(f"cnvrg_linechart_val_accuracy value: {val_accuracy:.4f}")

# Save the trained model
output_dir = "model_artifacts"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
