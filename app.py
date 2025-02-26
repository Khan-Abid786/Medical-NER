import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load the trained model and tokenizer
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"  # Update with your model path if different
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define label mapping
label_mapping = {
    "LABEL_0": "Non-medical",
    "LABEL_1": "Condition",
    "LABEL_2": "Disease"
}

# Initialize pipeline
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def predict_entities(text):
    predictions = nlp_pipeline(text)
    formatted_predictions = [
        {
            "Word": ent["word"],
            "Label": label_mapping.get(ent["entity_group"], ent["entity_group"]),
            "Confidence": f"{ent['score'] * 100:.2f}%"
        }
        for ent in predictions
    ]
    return formatted_predictions

# Streamlit UI
st.title("Medical NER using BioBERT")
st.write("Enter a medical text to extract conditions and diseases.")

# User Input
text_input = st.text_area("Enter text here:", "The patient was diagnosed with glioblastoma and prescribed temozolomide.")

if st.button("Analyze"):
    if text_input:
        results = predict_entities(text_input)
        st.write("### Predictions:")
        for result in results:
            st.write(f"**{result['Word']}** → {result['Label']} ({result['Confidence']})")
    else:
        st.warning("Please enter some text!")

st.sidebar.write("Model: BioBERT")
st.sidebar.write("Developed for Medical NER")
