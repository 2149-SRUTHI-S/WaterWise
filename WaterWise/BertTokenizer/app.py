from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
import pandas as pd

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load CSV with proper encoding
df = pd.read_csv("dataset/agricultural_water_footprint.csv", encoding="utf-8", usecols=[0, 1])  # Only first 2 columns

# Convert the data into a dictionary (Crop -> Water Use in Litres/kg)
water_footprint_data = {
    str(row[0]).lower().strip(): float(row[1]) // 100  # Convert m³ to Litres
    for row in df.itertuples(index=False)
}


def preprocess_text(text):
    """Normalize text input (lowercase, remove extra spaces/punctuation)."""
    return text.lower().strip().replace("!", "").replace("?", "").replace(",", "")

def tokenize_input(text):
    """Tokenize input text using BERT tokenizer."""
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=10, return_tensors="pt")
    return inputs["input_ids"].tolist()[0], inputs["attention_mask"].tolist()[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    product_name = data.get("product_name", "").strip()
    
    if not product_name:
        return jsonify({"error": "No product name provided"}), 400
    
    processed_text = preprocess_text(product_name)
    token_ids, attention_mask = tokenize_input(processed_text)
    
    # Find closest match in dataset (handling subword tokenization cases)
    matched_product = None
    for key in water_footprint_data.keys():
        if key in processed_text:
            matched_product = key
            break

    processed_text = matched_product
    
    water_footprint = water_footprint_data.get(matched_product, "Data not available") if matched_product else "Data not available"
    
    return jsonify({
        "original_input": product_name,
        "processed_text": processed_text,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "predicted_product": matched_product if matched_product else "Unknown",
        "water_footprint_liters": water_footprint
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
