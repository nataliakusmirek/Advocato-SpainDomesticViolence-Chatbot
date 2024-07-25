import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd
from googletrans import Translator


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
        self.translator = Translator()


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
        
        with open('definitions.txt', 'r', encoding='utf-8') as file:
            general_definitions = file.readlines()

        with open('english_form.txt', 'r', encoding='utf-8') as file:
            english_form_data = file.readlines()
        
        with open('spanish_form.txt', 'r', encoding='utf-8') as file:
            spanish_form_data = file.readlines()
        
        with open('violence_definitions.txt', 'r', encoding='utf-8') as file:
            violence_definitions = file.readlines()
        
        survey_data = pd.read_csv('survey.csv')

        return general_definitions, english_form_data, spanish_form_data, violence_definitions, survey_data
    


    def preprocess_text_data(definitions, english_form_data, spanish_form_data, violence_definitions, survey_data):
        """
        Preprocess text data provided for training purposes.

        Args:
        definitions: text file, general definitions
        english_form_data: text file, form data in English
        spanish_form_data: text file, form data in Spanish
        violence_definitions: text file, violence definitions
        survey_data: pd.DataFrame, survey data

        Returns:
        list, input_texts
        list, target_texts
        
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

        for row in survey_data:
            input_text = f"Incident reported on {row['Report Date']} in {row['County']} county by {row['Agency']}"
            target_text = f"Offense: {row['Offense']}, Offender: {row['Offender']} ({row['Offender Age']} years old, {row['Offender Gender']}, {row['Offender Race']}, {row['Offender Ethnicity']}), Victim: {row['Victim']} ({row['Victim Age']} years old, {row['Victim Gender']}, {row['Victim Race']}, {row['Victim Ethnicity']}), Relationship: {row['Victim/Offender Relationship']}, Year: {row['Report Year']}"
        
            input_texts.append(input_text)
            target_texts.append(target_text)

        return input_texts, target_texts
    
    input_texts, target_texts = preprocess_text_data(definitions, english_form_data, spanish_form_data, violence_definitions, survey_data)


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


    def response(self, input_text, language):
        """
        Generate a response based on the input text.

        Args:
        self: instance of the class
        input_text: str, input text
        language: str, language of the input text

        Returns:
        str, response text
            
        """
        processed_input = self.preprocess(input_text)
        response = self.model.predict(processed_input)
        response_text = self.vectorizer.get_vocabulary()[tf.argmax(response[0])]
        
        if language.lower() == 'spanish':
            response_text = self.translator.translate(response_text, src='en', dest='es').text
        
        return response_text
    
    
def main():
    # Get user input
    language = input("What language would you like to use? (English or Spanish) ")
    while True:
        user_input = input("How can I help you today? ")
        if user_input.lower() == 'exit':
            break

        chatbot = Chatbot(max_length=256, vocab_size=10000)

        # Load data
        general_definitions, english_form_data, spanish_form_data, violence_definitions, survey_data = chatbot.load_data()
        chatbot.preprocess_text_data(general_definitions, english_form_data, spanish_form_data, violence_definitions, survey_data)
        chatbot.adapt(chatbot.input_texts, chatbot.target_texts)

        # Train the model
        chatbot.train(chatbot.input_texts, chatbot.target_texts)

        response = chatbot.response(user_input, language)
        print(response)

if __name__ == "__main__":
    main()


# Deploy on Django site!