import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load the LLaMA model and tokenizer (replace this with LLaMA 3 if it's available)
st.title("LLaMA 3 Model in Streamlit")

# Loading the model and tokenizer (use appropriate model path)
model_name = "meta-llama/Llama-3.1-8B"  # Example: Replace with LLaMA 3 model if available
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Display the input prompt in the sidebar
input_text = st.text_area("Enter prompt for LLaMA", "Hello, LLaMA!")

# When button is clicked, generate output using the model
if st.button("Generate Text"):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Ensure model is in evaluation mode and on the correct device (use GPU if available)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Show the result on the Streamlit app
    st.subheader("Generated Text:")
    st.write(generated_text)
