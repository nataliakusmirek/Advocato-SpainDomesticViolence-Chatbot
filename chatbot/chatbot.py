import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd

class Chatbot:
    def __init__(self, max_length, vocab_size):
        """
        Initialize the class.
        
        Args:
        self: instance of the class
        max_length: int, maximum length of input sequence
        vocab_size: int, size of the vocabulary

        Returns:
        None

        """
        super().__init__()
        self.max_length = max_length
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=max_length)
        self.model = self.build_model(vocab_size, max_length)


    def build_model(self, vocab_size, max_length):
        """
        Build the model.

        Args:
        self: instance of the class
        vocab_size: int, size of the vocabulary
        max_length: int, maximum length of input sequence

        Returns:
        model, instance of the model

        """
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(128, activation='relu'),
            Dense(vocab_size, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    

    def preprocess(self, inputs):
        """
        Preprocess the inputs and return the tokenized inputs.

        Args:
        self: instance of the class
        inputs: str, input text

        Returns:
        dict, tokenized inputs
        
        """
        inputs = '[CLS] ' + inputs + ' [SEP]'
        tok_inputs = self.vectorizer(inputs)

        return {
            "input_ids": tok_inputs,
            "input_mask": tf.cast(tok_inputs > 0, tf.int32),
            "segment_ids" : tf.zeros_like(tok_inputs)
        }
    
        
    def adapt(self, data):
        """
        Adapt the vectorizer to the data.

        Args:
        self: instance of the class
        data: list, input data

        Returns:
        None

        """
        self.vectorizer.adapt(data)


    def load_data():
        """
        Load the data from the files.

        Args:
        None

        Returns:
        pd.DataFrame, general definitions
        pd.DataFrame, form data
        pd.DataFrame, violence definitions
        pd.DataFrame, survey data
            
        """
        # Note: survey data was already cleaned and null values removed before access was granted
        general_definitions = pd.read_csv('definitions.txt')
        form_data = pd.read_csv('form.txt')
        violence_definitions = pd.read_csv('violence_definitions.txt')
        survey_data = pd.read_csv('fin.csv')

        return general_definitions, form_data, violence_definitions, survey_data
    
    def train(self, inputs, outputs, epochs=100):
        """
        Train the model.

        Args:
        self: instance of the class
        inputs: list, input data
        outputs: list, output data
        epochs: int, number of epochs

        Returns:
        None
            
        """
        X = self.vectorizer(inputs)
        y = self.vectorizer(outputs)
        self.model.fit(X, y, epochs=epochs)

    def response(self, input_text):
        """
        Generate a response based on the input text.

        Args:
        self: instance of the class
        input_text: str, input text

        Returns:
        str, response text
            
        """
        processed_input = self.preprocess(input_text)
        response = self.model.predict(processed_input)
        response_text = self.vectorizer.get_vocabulary()[tf.argmax(response[0])]
        return response_text
    
def main():
    # Get user input
    language = input("What language would you like to use? (English or Spanish) ")
    user_input = input("How can I help you today? ")

    chatbot = Chatbot(max_length=256, vocab_size=10000)

    # Load data
    general_definitions, form_data, violence_definitions, survey_data = chatbot.load_data()
    all_data = pd.concat([general_definitions, form_data, violence_definitions, survey_data])
    chatbot.adapt(all_data['text'])

    # Train the model
    #???

    # Generate response
    response = chatbot.response(user_input)
    print(response)

if __name__ == "__main__":
    main()


# Implement multilingual support for Spanish users

# Provide a little "dictionary" feature of the chatbot of legal terms and definitions

# Deploy on Django site!