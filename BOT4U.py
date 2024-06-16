import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFaceEndpoint
import os

# Instantiate the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# Define a mapping dictionary for sentiment labels
label_map = {
    '5 stars': 'Positive',
    '4 stars': 'Positive',
    '3 stars': 'Neutral',
    '2 stars': 'Negative',
    '1 star': 'Negative'
}

# Set your Hugging Face API token
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Ensure API key is set properly
if not api_key:
    raise ValueError("Hugging Face API token not found. Set the environment variable 'HUGGINGFACEHUB_API_TOKEN'.")

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
            # Perform sentiment analysis
            result = classifier(sentence)
            # Extract sentiment label and map to 'Positive' or 'Negative'
            sentiment_label = result[0]['label']
            sentiment = label_map.get(sentiment_label, 'Unknown')
            # Display result
            st.write(f"The sentiment of the sentence '{sentence}' is: {sentiment}")