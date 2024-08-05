import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras import Model
import numpy as np
import config

class PositionalEncoding(Layer):
    """
    Positional encoding layer for the Transformer model. 
    This layer adds positional information to the input embeddings.
    """
    def __init__(self, max_positional_encoding, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.get_positional_encoding(max_positional_encoding, d_model)

    def get_positional_encoding(self, max_positional_encoding, d_model):
        angles = np.arange(max_positional_encoding)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
        angles = np.concatenate([np.sin(angles[:, ::2]), np.cos(angles[:, 1::2])], axis=-1)
        positional_encoding = tf.constant(angles[np.newaxis, ...], dtype=tf.float32)
        return positional_encoding

    def call(self, x):
        return x + self.positional_encoding[:, :tf.shape(x)[1], :]

class MultiHeadAttention(Layer):
    """
    Multi-head attention layer for the Transformer model.
    This layer performs multiple attention calculations in parallel, to allow the model to focus on different parts of the input sequence.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]

        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(value), batch_size)

        # Scaled dot-product attention
        scaled_attention = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention += (mask * -1e9)
        scaled_attention = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(scaled_attention, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(concat_attention)

class TransformerBlock(Layer):
    """
    Transformer block layer for the Transformer model.
    This layer contains a multi-head attention layer and a feedforward neural network layer, with residual connections and layer normalization applied to each sublayer.
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Encoder(Layer):
    """
    Encoder layer for the Transformer model.
    This layer contains multiple Transformer block layers to process the input sequence.
    """
    def __init__(self, num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_dim=10000, output_dim=d_model)
        self.pos_encoding = PositionalEncoding(max_positional_encoding, d_model)
        self.enc_layers = [TransformerBlock(d_model, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

class Decoder(Layer):
    """
    Decoder layer for the Transformer model.
    This layer contains multiple Transformer block layers to process the output sequence.
    """
    def __init__(self, num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_dim=10000, output_dim=d_model)
        self.pos_encoding = PositionalEncoding(max_positional_encoding, d_model)
        self.dec_layers = [TransformerBlock(d_model, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, look_ahead_mask)
        return x

class Transformer(Model):
    """
    Transformer model for sequence-to-sequence tasks.
    This model contains an encoder and a decoder, with an embedding layer and a final linear layer to convert the decoder output to a prediction.
    """
    def __init__(self, num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate)
        self.final_layer = Dense(config.VOCAB_SIZE)

    def call(self, enc_input, dec_input, training, look_ahead_mask, padding_mask):
        enc_output = self.encoder(enc_input, training, padding_mask)
        dec_output = self.decoder(dec_input, enc_output, training, look_ahead_mask, padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output


def build_model():
    model = Transformer(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        ff_dim = config.UNITS,
        max_positional_encoding=config.MAX_LENGTH,
        dropout_rate=config.DROPOUT)
    return model