import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, TextVectorization
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from googletrans import Translator

# Note: loss is ~0.22, accuracy is ~0.92

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
        self.vocab_size = vocab_size
        self.vectorizer = CountVectorizer(max_features=vocab_size, tokenizer=lambda x: x.split())
        self.model = self.build_model(vocab_size)
        self.translator = Translator()
        self.index_word = {}

    def build_model(self, vocab_size):
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
            tf.keras.Input(shape=(297,)),
            Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
            LSTM(128, return_sequences=True),
            LSTM(128, return_sequences=True),
            Dense(vocab_size, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model



    def adapt(self, data):
        """
        Adapt the vectorizer to the data.

        Args:
        self: instance of the class
        data: list, input data

        Returns:
        None

        """
        self.vectorizer.fit(data)
        self.index_word = {i: word for word, i in self.vectorizer.vocabulary_.items()}


    def load_data(self):
        """
        Load the data from the files.

        Args:
        self: instance of the class

        Returns:
        tuple: (general_definitions, english_form_data, spanish_form_data, violence_definitions)
            
        """
        # Note: survey data was already cleaned and null values removed before access was granted
        
        with open('training_sets/definitions.txt', 'r', encoding='utf-8') as file:
            general_definitions = file.readlines()

        with open('training_sets/english_form.txt', 'r', encoding='utf-8') as file:
            english_form_data = file.readlines()
        
        with open('training_sets/spanish_form.txt', 'r', encoding='utf-8') as file:
            spanish_form_data = file.readlines()
        
        with open('training_sets/violence_definitions.txt', 'r', encoding='utf-8') as file:
            violence_definitions = file.readlines()
        
        return general_definitions, english_form_data, spanish_form_data, violence_definitions
    


    def preprocess_text_data(self, definitions, english_form_data, spanish_form_data, violence_definitions):
        """
        Preprocess text data provided for training purposes.

        Args:
        self: instance of the class
        definitions: text file, general definitions
        english_form_data: text file, form data in English
        spanish_form_data: text file, form data in Spanish
        violence_definitions: text file, violence definitions

        Returns:
        tuple: (input_texts, target_texts)
        
        """
        input_texts = []
        target_texts = []

        for line in definitions:
            if ':' in line:
                term, definition = line.split(':', 1)
                input_texts.append(term)
                target_texts.append(definition.strip())

        for line in violence_definitions:
            if ':' in line:
                term, definition = line.split(':', 1)
                input_texts.append(term)
                target_texts.append(definition.strip())

        for line in english_form_data:
            if ':' in line:
                section, details = line.split(':', 1)
                input_texts.append(section)
                target_texts.append(details.strip())

        for line in spanish_form_data:
            if ':' in line:
                section, details = line.split(':', 1)
                input_texts.append(section)
                target_texts.append(details.strip())

        print(f"Total input texts: {len(input_texts)}")
        print(f"Total target texts: {len(target_texts)}")

        if len(input_texts) == 0 or len(target_texts) == 0:
            raise ValueError("Input texts or target texts are empty after preprocessing.")

        # Vectorize data without printing it
        self.adapt(input_texts + target_texts)
        X = self.vectorizer.transform(input_texts).toarray()
        y = self.vectorizer.transform(target_texts).toarray()

        return X, y


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
        # Convert lists to tensors
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)

        # Reshape outputs for sparse_categorical_crossentropy
        outputs = tf.expand_dims(outputs, -1)  # Shape: (batch_size, sequence_length, 1)

        print("Shape of inputs:", inputs.shape)
        print("Shape of outputs:", outputs.shape)

        self.model.fit(inputs, outputs, epochs=epochs, batch_size=1)



    def preprocess(self, text):
        """
        Preprocess the input text.

        Args:
        self: instance of the class
        text: str, input text

        Returns:
        tensor, processed input text
            
        """
        vectorized_input = self.vectorizer.transform([text]).toarray()
        # Ensure the shape is (1, 297)
        processed_input = vectorized_input.reshape(1, -1)
        if processed_input.shape[0] != 1 or processed_input.shape[1] != 297:
            processed_input = tf.squeeze(processed_input, axis=1)
        return processed_input


    def response(self, user_input, language):
        """
        Generate a response from the model based on user input and selected language.
        Args:
        self: instance of the class
        user_input: str, input from user
        language: str, selected language
        """
        if language.lower() == 'spanish':
            processed_input = self.preprocess_spanish_input(user_input)
        else:
            processed_input = self.preprocess_english_input(user_input)

        # Convert to tensor
        processed_input = tf.convert_to_tensor(processed_input, dtype=tf.float32)
    
        if len(processed_input.shape) > 2:
            processed_input = tf.squeeze(processed_input, axis=1)

        print("Shape of processed input:", processed_input.shape)
        prediction = self.model.predict(processed_input)
        response_text = self.postprocess_prediction(prediction)

        return response_text


    def preprocess_spanish_input(self, text):
        """
        Preprocess Spanish text input.
        Args:
        self: instance of the class
        text: str, input text in Spanish
        """
        # Vectorize and pad the Spanish text input
        translated_text = self.translator.translate(text, src='es', dest='en').text
        vectorized_input = self.vectorizer.transform([translated_text]).toarray()
        # Ensure the shape is (1, 297)
        processed_input = vectorized_input.reshape(1, -1)
        return processed_input

    def preprocess_english_input(self, text):
        """
        Preprocess English text input.
        Args:
        self: instance of the class
        text: str, input text in English
        """
        vectorized_input = self.vectorizer.transform([text]).toarray()
        # Ensure the shape is (1, 297)
        processed_input = vectorized_input.reshape(1, -1)
        return processed_input

    def postprocess_prediction(self, prediction):
        """
        Convert model prediction to human-readable text.
        Args:
        self: instance of the class
        prediction: model prediction
        """
        # Convert prediction to text
        predicted_indices = tf.argmax(prediction, axis=-1).numpy().flatten()
        response_words = [self.index_word.get(index, '<UNK>') for index in predicted_indices]
        response_text = ' '.join(response_words).strip()
        return response_text

    
    
def main():
    print("Initializing the chatbot...")
    # Initialize the chatbot
    chatbot = Chatbot(max_length=256, vocab_size=10000)

    print("Loading data...")
    # Load data
    general_definitions, english_form_data, spanish_form_data, violence_definitions = chatbot.load_data()
    print("Data loaded.")

    print("Preprocessing text data...")
    # Preprocess text data
    input_texts, target_texts = chatbot.preprocess_text_data(general_definitions, english_form_data, spanish_form_data, violence_definitions)
    
    print("Adapting vectorizer...")
    print("Text data preprocessed and adapted to vectorizer.")

    print("Training the model...")
    # Train the model
    chatbot.train(input_texts, target_texts)
    print("Model trained.")
    # Get user input
    language = input("What language would you like to use? (English or Spanish) ")
    if language.lower() not in ['english', 'spanish']:
        language = input("Please enter either English or Spanish as your preferred language: ")
        return
    
    print(f"Language selected: {language}")

    while True: 
        user_input = input("How can I help you today? ")
        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye', 'stop', 'end']:
            print("Goodbye!")
            break
        
        # Generate a response
        response = chatbot.response(user_input, language)
        print(response)

if __name__ == "__main__":
    main()


# Deploy on Django site!