import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model_name = "/cnvrg/model_artifacts"  
# Replace with the path to your saved model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the classification labels
labels = ['Login Issues', 'Payment Problems', 'Technical Issues', 'Installation Help', 'Order Inquiry', 'Billing Issues', 'Email Problems']  # Replace with your own labels

# Define the predict function
def predict(input_text):
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=256
    )

    # Move the inputs to the appropriate device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform the forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted label
    predicted_label = labels[torch.argmax(outputs.logits)]

    return predicted_label

# Define the inference route
@app.route('/predict', methods=['POST'])
def handle_prediction():
    # Get the text from the request
    input_text = request.json['input_params']

    # Perform prediction
    predicted_label = predict(input_text)

    # Return the predicted label as JSON response
    return jsonify({'label': predicted_label})

# Run the Flask application
if __name__ == '__main__':
    app.run()
