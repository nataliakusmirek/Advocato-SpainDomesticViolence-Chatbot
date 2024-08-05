# Built with the support of blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
from model import Model
import tensorflow as tf
import numpy as np
import data
from model import build_model
from model import Model
import os
import config
from googletrans import Translator

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        """
        Custom learning rate schedule for the Transformer model.
        This schedule increases the learning rate linearly for the first warmup_steps training steps, then decreases it proportionally to the inverse square root of the step number.
        This helps to improve training speed and performance, specifically the convergence of the model.
        """
        def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()
                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)
                self.warmup_steps = warmup_steps

        def __call__(self, step):
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)        

def load_data():
        try:
                # Load definitions and conversations
                print("Loading data...")
                id2def = data.get_definitions()
                print(f"Definitions loaded: {len(id2def)} entries")
        
                convos = data.get_conversations()
                print(f"Conversations loaded: {len(convos)} entries")
        
                # Build subsets of definitions and conversations
                print("Building subsets...")
                definitions, conversations = data.build_subsets(id2def, convos)
                print(f"Definitions and conversations subsets built: {len(definitions)} entries each")
        
                # Build and save datasets
                print("Building and saving datasets...")
                data.build_datasets(definitions, conversations)
                print("Datasets built and saved")
        
                # Build tokenizer
                print("Building tokenizer...")
                tokenizer = data.build_tokenizer(definitions, conversations)
                print(f"Tokenizer built with {len(tokenizer.word_index)} tokens")

                # Set VOCAB_SIZE in config
                config.VOCAB_SIZE = len(tokenizer.word_index) + 1  # Adding 1 for padding token
                print(f"Vocabulary size set to {config.VOCAB_SIZE} for model configuration")

                # Tokenize and pad sequences
                print("Tokenizing and padding sequences...")
                tokenized_defs, tokenized_convos = data.tokenize_tokens(tokenizer, definitions, conversations)
                print(f"Tokenized and padded definitions and conversations")
        
                # Filter out long sequences
                print("Filtering long sequences...")
                filtered_defs, filtered_convos = data.filter_long_sequences(tokenized_defs, tokenized_convos)
                print(f"Filtered sequences: {len(filtered_defs)} definitions, {len(filtered_convos)} conversations")

                # Prepare the data for training
                print("Preparing training data...")
                enc_inputs, dec_inputs, dec_targets = data.prepare_training_data(filtered_defs, filtered_convos)
                print(f"Training data prepared: {len(enc_inputs)} encoder inputs, {len(dec_inputs)} decoder inputs, {len(dec_targets)} decoder targets")
    
                # Build TensorFlow dataset
                print("Building TensorFlow dataset...")
                train_data, val_data, test_data = data.build_tf_datasets(filtered_defs, filtered_convos)
                print("TensorFlow datasets built and saved in 'datasets' folder")
        
                print("Data preparation complete.")
                return train_data, val_data, test_data, tokenizer
    
        except Exception as e:
                print(f"An error occurred: {e}")

def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
       y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))
       accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
       return accuracy


class Chatbot:
        """
        Chatbot class for the Transformer model.
        This class contains methods for creating the model, training the model, evaluating the model on test data, getting user input, and building a response.
        """
        def __init__(self):
                print("Initializing chatbot...")
                self.model = build_model()
                self.optimizer = tf.keras.optimizers.Adam(CustomSchedule(config.D_MODEL))
                self.model.compile(optimizer=self.optimizer, loss=loss, metrics=[accuracy])
                self.train_data, self.val_data, self.test_data, self.tokenizer = load_data()
                self.translator = Translator()
                print("Chatbot initialized.")

        def train(self, epochs=100):
                print("Starting training...")
                history = self.model.fit(
                        [self.train_data[0], self.train_data[1]],  # encoder input and decoder input
                        self.train_data[2],  # decoder target
                        epochs=epochs,
                        validation_data=([self.val_data[0], self.val_data[1]], self.val_data[2])
                )
                print("Training complete.")
                return history
        
        def evaluate_test(self):
                print("Evaluating test data...")
                test_loss, test_accuracy = self.model.evaluate(
                        [self.test_data[0], self.test_data[1]],  # encoder input and decoder input
                        self.test_data[2]  # decoder target
                )
                print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

        def get_user_input(self):
                user_input = input("User: ")
                return user_input

        def preprocess_sentence(self, sentence):
                print(f"Preprocessing sentence: {sentence}")
                sentence = sentence.lower().strip()
                sentence = self.translator.translate(sentence, src='es', dest='en').text
                sequence = self.tokenizer.texts_to_sequences([sentence])
                sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=config.MAX_LENGTH, padding='post')
                print(f"Preprocessed sequence: {sequence}")
                return sequence
        
        def generate_response(self, model_output):
                print(f"Generating response...")
                sequence = tf.argmax(model_output, axis=-1)
                response = ' '.join([word for word in sequence[0].numpy() if word != 0])
                print(f"Generated response: {response}")
                return response
        
        def build_response(self, user_input):
                print(f"Building response for user input: {user_input}")
                sequence = self.preprocess_sentence(user_input)
                enc_input = sequence
                dec_input = sequence[:, :-1]
                dec_target = sequence[:, 1:]
                model_output = self.model.predict([enc_input, dec_input])
                response = self.generate_response(model_output)
                return response
                

        def translate_spanish_output(self, final_response):
                print(f"Translating response to Spanish: {final_response}")
                response = self.translator.translate(final_response, src='en', dest='es').text
                print(f"Translated response: {response}")
                return response


        def chat(self):
                print("Chatbot is ready to chat! Type 'exit' to end the conversation.")
                language_type = input("English or Spanish? ")
                while True:
                        user_input = self.get_user_input()
                        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye', 'stop', 'end', 'done', 'finish', 'finished']:
                                print("Chatbot: Goodbye!")
                                break
                        response = self.build_response(user_input)
                        if language_type.lower() == 'spanish' or language_type.lower() == 'espanol':
                                response = self.translate_spanish_output(response)
                                print(f"Chatbot: {response}") # For Spanish users
                        else:
                                print(f"Chatbot: {response}") # For English users
                


def main():
        chatbot = Chatbot()
        chatbot.train()
        chatbot.evaluate_test()
        chatbot.chat()

if __name__ == '__main__':
        main()

