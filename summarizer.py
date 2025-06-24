import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Summarization function
def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=summary_length,
        min_length=min(summary_length, 80),
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.set_page_config(page_title="AI Text Summarizer", layout="wide")
st.title("📝 AI Text Summarizer")
input_text = st.text_area("Enter your text here:", height=300)
summary_length = st.slider("Select summary length:", 30, 300, 100, step=10)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_text(input_text)
        st.subheader("📌 Summary:")
        st.write(summary)
