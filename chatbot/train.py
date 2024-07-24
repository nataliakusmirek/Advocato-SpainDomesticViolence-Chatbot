# train to help complete form
# train to respond in english and spanish (should be first question before chatbot helps any further
# 
# Features:
# Multilingual support (English and Spanish).
# Form completion guidance.
# Definition explanations for legal terms.
# Risk assessment and support resource information.

import tensorflow as tf
import pandas as pd


def get_input():
    language = input("What language would you like to use? (English or Spanish) ")
    user_input = input("How can I help you today? ")
    tokenize_input(language)
    tokenize_input(user_input)

def tokenize_input(input):

    # Tokenize user input and remove stop words, then perform stemming/lemmatization

# Load training data
def load_data():
    general_definitions = pd.read_csv('definitions.txt')
    form_data = pd.read_csv('form.txt')
    violence_definitions = pd.read_csv('violence_definitions.txt')
    survey_data = pd.read_csv('fin.csv')

# Label training data based on intent
def fill_form():
    pass

def ask_definition():
    pass

def assess_risk():
    pass

def provide_support():
    pass

# Use transformers to train model Tensorflow for building the model, Sci-Kit Learn for additional preprocessing

# Use gradient descent to minimize loss function and possibly Grid Search for hyperparameter tuning




# Implement multilingual support for Spanish users

# Provide a little "dictionary" feature of the chatbot of legal terms and definitions

# Deploy on Django site!