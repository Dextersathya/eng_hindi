import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_path = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)


# Function to translate text
def translate(text):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200,  # Adjust this value according to your text length
    )

    # Perform the translation
    translated = model.generate(**inputs, max_length=200)

    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text


# Streamlit UI
st.title("English to Hindi Translation")

# Input text box for large text
input_text = st.text_area("Enter English text for translation", height=200)

# Translate button
if st.button("Translate"):
    if input_text:
        translated_text = translate(input_text)
        st.write("Translated Hindi Text:")
        st.write(translated_text)
    else:
        st.write("Please enter some text for translation.")
