import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# Load model and tokenizer
MODEL_PATH = "model"  # Path to the model folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Create NER pipeline
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Streamlit UI
st.title("🩺 Medical Named Entity Recognition (NER)")
st.write("This model identifies medical conditions and diseases in text.")

# User Input
user_input = st.text_area("Enter medical text here:", "")

if st.button("Analyze"):
    if user_input:
        results = nlp_pipeline(user_input)
        
        # Display results
        st.subheader("Predicted Entities:")
        for entity in results:
            st.write(f"**{entity['word']}** → `{entity['entity_group']}` ({entity['score']:.2%})")
    else:
        st.warning("Please enter some text.")

# Footer
st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io/) & [Hugging Face Transformers](https://huggingface.co/).")
