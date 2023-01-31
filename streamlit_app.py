import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelWithLMHead.from_pretrained('gpt2')

st.title('GPT-3 Streamlit Generator')
user_input = st.text_input('Input your text here', value='')

# Generating response with GPT-3
generated_response = model.generate(input_ids=tokenizer.encode(user_input), max_length=50)

# Output the generated response
st.markdown(tokenizer.decode(generated_response[0]))
