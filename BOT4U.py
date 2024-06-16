import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from langchain_community.llms import HuggingFaceEndpoint
import os

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'api key'

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# Create the language model endpoint
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.9,
    max_new_tokens=500
)

st.title('BOT4U')

# Sidebar menu for navigation
selected_section = st.sidebar.radio("Navigation", ["Text Generation", "Sentiment Analysis"])

if selected_section == "Text Generation":
    st.header('Text Generation')

    prompt = st.text_input("Enter your prompt")

    if prompt:
        generated_text = llm(prompt)
        st.write(generated_text)

elif selected_section == "Sentiment Analysis":
    st.header('Sentiment Analysis')

    sentence = st.text_area("Enter a sentence for sentiment analysis")

    if st.button("Analyze"):
        if sentence:
            # Tokenize input text
            inputs = tokenizer(sentence, return_tensors='pt')

            # Perform sentiment analysis
            with torch.no_grad():
                outputs = model(**inputs)

            # Get predicted sentiment (0: negative, 1: positive)
            prediction = torch.argmax(outputs.logits, dim=1).item()

            # Display result
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"The sentiment of the sentence '{sentence}' is: {sentiment}")