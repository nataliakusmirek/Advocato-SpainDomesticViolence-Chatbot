"""
Load, preprocess, and tokenize training and testing data for model.

"""

def load_file_data():
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


def preprocess_text_data(definitions, english_form_data, spanish_form_data, violence_definitions):
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

        return input_texts, target_texts


# complete these functions
def tokenizer(input_texts, target_texts, normalize_digits=True):
    for line in input_texts:
        pass
    
    for line in target_texts:
        pass

def build_vocab():
     pass

def load_vocab():
     pass

def sentence2id():
     pass

def token2id():
     pass

def process_data():
     # build_vocab and then token2id each (encode and decode)
     pass

def load_model_data():
     pass



## consider whether we need padding, reshaping, batching?

if __name__ == '__main__':
    definitions, english_form_data, spanish_form_data, violence_definitions = load_file_data()
    preprocess_text_data(definitions, english_form_data, spanish_form_data, violence_definitions)


