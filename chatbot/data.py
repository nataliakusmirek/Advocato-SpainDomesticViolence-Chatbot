# Built with the support of blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
import numpy as np
import tensorflow as tf
import os
import re
import random
import json

# Dataset format
# LINE_ID + ++ + SUBJECT_ID + ++ + DEFINITION_ID + ++ + DEFINITION_TEXT

def get_definitions():
    """
    Extracts text from the definitions.txt file and returns a dictionary of id to definition
    """
    id2def = {}
    with open('data/definitions.txt', 'r') as file:
        i = 0
        try:
            for line in file:
                parts = line.strip().split(' + ++ + ')
                if len(parts) == 4:
                    key = parts[0]
                    definition = parts[3]
                    id2def[key] = definition
                i += 1
        except UnicodeDecodeError:
            print(f'Error at line {i}: {line}')
    return id2def


def get_conversations():
    """
    Extracts text from the conversations.txt file and returns a list of conversations
    """
    convos = []
    with open('data/conversations.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(' + ++ + ')
            if len(parts) == 4:
                key = parts[0]
                convo = parts[3]
                convos.append((key, convo))
    return convos


def build_subsets(id2def, convos):
    """
    Builds the dataset into two sets: definitions and conversations the chatbot will have
    """
    definitions, conversations = [], []
    for key, convo in convos:
        if key in id2def:
            definitions.append(id2def[key])
            conversations.append(convo)
        else:
            print(f"Key '{key}' not found in id2def")
    assert len(definitions) == len(conversations)
    return definitions, conversations


def build_datasets(definitions, conversations, test_ratio=0.2, val_ratio=0.1):
    """
    Builds the training and testing datasets from the definitions and conversations sets
    """
    num_entries = len(definitions)
    test_size = int(num_entries * test_ratio)
    val_size = int(num_entries * val_ratio)

    indices = list(range(num_entries))
    random.shuffle(indices) # Shuffles indices to ensure randomness in splitting

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    filenames = ['train.enc', 'train.dec', 'val.enc', 'val.dec', 'test.enc', 'test.dec']
    files = []

    for filename in filenames:
        files.append(open(os.path.join('dataset', filename), 'w'))

    for i in range(num_entries):
        if i in test_indices:
            files[4].write(definitions[i] + '\n')
            files[5].write(conversations[i] + '\n')
        elif i in val_indices:
            files[2].write(definitions[i] + '\n')
            files[3].write(conversations[i] + '\n')
        else:
            files[0].write(definitions[i] + '\n')
            files[1].write(conversations[i] + '\n')
            
    for file in files:
        file.close()


def load_vocab(vocab_file='chatbot/vocab.json'):
    """
    Loads the vocabulary from the vocab.json file
    """
    with open(vocab_file, 'r') as file:
        word_index = json.load(file)
    return word_index


def add_special_tokens(word_index):
    """
    Adds special tokens to the vocabulary for better training
    """
    special_tokens = {
        '<pad>' : 0,
        '<unk>' : 1,
        '<s>' : 2,
        '</s>' : 3
    }
    word_index = {**special_tokens, **word_index}

    # Update and save vocabulary
    with open('chatbot/vocab_with_special_tokens.json', 'w') as file:
        json.dump(word_index, file)
    return word_index


def build_tokenizer(definitions, conversations):
    """
    Builds a tokenizer for the definitions and conversations
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(definitions + conversations)

    word_index = tokenizer.word_index
    word_index = add_special_tokens(word_index)

    # Save the tokenizer
    with open('vocab.json', 'w') as file:
        json.dump(word_index, file)
        
    return tokenizer


def tokenize_tokens(tokenizer, definitions, conversations):
    """
    Tokenizes each definition and conversation
    """
    word_index = tokenizer.word_index
    start_token = tokenizer.word_index.get('<s>', 2)
    end_token = tokenizer.word_index.get('</s>', 3)
    unk_token = tokenizer.word_index.get('<unk>', 1)

    def add_start_end_tokens(texts):
        """
        Adds start and end tokens to each text
        """
        sequences = tokenizer.texts_to_sequences(texts)
        for i in range(len(sequences)):
            sequences[i] = [start_token] + sequences[i] + [end_token]
            sequences[i] = [token if token in word_index.values() else unk_token for token in sequences[i]]

        return sequences
    
    def pad_sequences_with_tokens(conversations):
        """
        Tokenizes text data and pads sequences to ensure uniformity
        """
        sequences = add_start_end_tokens(conversations) 
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            padding='post'
        )
    
    tokenized_defs = pad_sequences_with_tokens(definitions)
    tokenized_convos = pad_sequences_with_tokens(conversations)
    return tokenized_defs, tokenized_convos


def filter_long_sequences(tokenized_defs, tokenized_convos, max_length=100):
    """
    Filters out sequences that are too long
    """
    filtered_defs, filtered_convos = [], []

    for i in range(len(tokenized_defs)):
        if len(tokenized_defs[i]) <= max_length and len(tokenized_convos[i]) <= max_length:
            filtered_defs.append(tokenized_defs[i])
            filtered_convos.append(tokenized_convos[i])

    return filtered_defs, filtered_convos


def build_tf_datasets(filtered_defs, filtered_convos, batch_size=64, val_ratio=0.1, test_ratio=0.2):
    """
    Builds training, validation, and testing datasets by creating tf.data.Dataset objects.
    """
    num_entries = len(filtered_defs)
    test_size = int(num_entries * test_ratio)
    val_size = int(num_entries * val_ratio)

    # Shuffle and split dataset
    indices = np.arange(num_entries)
    np.random.shuffle(indices)
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    # Split the indices into training, validation, and testing (slicing)
    train_defs = np.array(filtered_defs)[train_indices]
    train_convos = np.array(filtered_convos)[train_indices]
    val_defs = np.array(filtered_defs)[val_indices]
    val_convos = np.array(filtered_convos)[val_indices]
    test_defs = np.array(filtered_defs)[test_indices]
    test_convos = np.array(filtered_convos)[test_indices]

    # Create tf.data.Dataset objects (TensorFlow datasets)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_defs, train_convos))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_defs, val_convos))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_defs, test_convos))

    # Shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)


    train_dataset.save('saved_datasets/training_tf_dataset')
    val_dataset.save('saved_datasets/validation_tf_dataset')
    test_dataset.save('saved_datasets/testing_tf_dataset')
    
    return train_dataset, val_dataset, test_dataset

def main():
    try:
        # Load definitions and conversations
        print("Loading data...")
        id2def = get_definitions()
        print(f"Definitions loaded: {len(id2def)} entries")
        
        convos = get_conversations()
        print(f"Conversations loaded: {len(convos)} entries")
        
        # Build subsets of definitions and conversations
        print("Building subsets...")
        definitions, conversations = build_subsets(id2def, convos)
        print(f"Definitions and conversations subsets built: {len(definitions)} entries each")
        
        # Build and save datasets
        print("Building and saving datasets...")
        build_datasets(definitions, conversations)
        print("Datasets built and saved")
        
        # Build tokenizer
        print("Building tokenizer...")
        tokenizer = build_tokenizer(definitions, conversations)
        print(f"Tokenizer built with {len(tokenizer.word_index)} tokens")
        
        # Tokenize and pad sequences
        print("Tokenizing and padding sequences...")
        tokenized_defs, tokenized_convos = tokenize_tokens(tokenizer, definitions, conversations)
        print(f"Tokenized and padded definitions and conversations")
        
        # Filter out long sequences
        print("Filtering long sequences...")
        filtered_defs, filtered_convos = filter_long_sequences(tokenized_defs, tokenized_convos)
        print(f"Filtered sequences: {len(filtered_defs)} definitions, {len(filtered_convos)} conversations")
        
        # Build TensorFlow dataset
        print("Building TensorFlow dataset...")
        dataset = build_tf_datasets(filtered_defs, filtered_convos)
        print("TensorFlow datasets built and saved in 'datasets' folder")
        
        print("Data preparation complete.")
        return dataset
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
