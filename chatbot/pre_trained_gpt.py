from transformers import GPT2Tokenizer, GPT2LMHeadModel
import tensorflow as tf

model_name = 'Advocato'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Get user input
def get_input():
    language = input("What language would you like to use? (English or Spanish) ")
    user_input = input("How can I help you today? ")

# Tokenize user input and remove stop words, then perform stemming/lemmatization
def generate_response(input):
    inputs = tokenizer.encode(input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=250, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)