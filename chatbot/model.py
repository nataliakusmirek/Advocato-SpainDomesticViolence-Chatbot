import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras import Model
import numpy as np
import config

class PositionalEncoding(Layer):
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
        print(f"Query shape: {query.shape}")
        print(f"Key shape: {key.shape}")
        print(f"Value shape: {value.shape}")
        if mask is not None:
            print(f"Mask shape before reshape: {mask.shape}")
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        print(f"Query shape after split_heads: {query.shape}")
        print(f"Key shape after split_heads: {key.shape}")
        print(f"Value shape after split_heads: {value.shape}")

        if mask is not None:
            mask = tf.reshape(mask, (batch_size, 1, 1, -1))
            print(f"Mask shape after reshape: {mask.shape}")
        
        attention_output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        print(f"Attention output shape after transpose and reshape: {attention_output.shape}")
        
        return self.dense(attention_output)
    
    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            mask = tf.cast(mask, dtype=scaled_attention_logits.dtype)
            mask = tf.reshape(mask, [tf.shape(mask)[0], 1, 1, tf.shape(mask)[-1]])
            scaled_attention_logits += (mask * -1e9)

        print(f"Mask shape in scaled_dot_product_attention: {mask.shape}")
        print(f"Scaled attention logits shape: {scaled_attention_logits.shape}")

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Output shape from scaled_dot_product_attention: {output.shape}")
        
        return output, attention_weights

class TransformerBlock(Layer):
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

    def call(self, x, enc_output=None, training=False, look_ahead_mask=None, padding_mask=None):
        attn_output = self.att(x, x, x, padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_dim=10000, output_dim=d_model)
        self.pos_encoding = PositionalEncoding(max_positional_encoding, d_model)
        self.enc_layers = [TransformerBlock(d_model, num_heads, ff_dim, dropout_rate) for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, look_ahead_mask=None, padding_mask=mask)
        return x

class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_dim=10000, output_dim=d_model)
        self.pos_encoding = PositionalEncoding(max_positional_encoding, d_model)
        self.dec_layers = [TransformerBlock(d_model, num_heads, ff_dim, dropout_rate) for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        return x

class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, ff_dim, max_positional_encoding, dropout_rate)
        self.final_layer = Dense(config.VOCAB_SIZE)

    def call(self, encoder_input, decoder_input, training=False, look_ahead_mask=None, padding_mask=None):
        print(f"encoder_input shape: {encoder_input.shape}")
        print(f"decoder_input shape: {decoder_input.shape}")

        enc_output = self.encoder(encoder_input, training=training, mask=padding_mask)
        print(f"Encoder output shape: {enc_output.shape}")

        dec_output = self.decoder(decoder_input, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        print(f"Decoder output shape: {dec_output.shape}")

        final_output = self.final_layer(dec_output)
        print(f"Final output shape: {final_output.shape}")
        
        return final_output


def build_model():
    model = Transformer(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        ff_dim=config.UNITS,
        max_positional_encoding=config.MAX_LENGTH,
        dropout_rate=config.DROPOUT)
    return model
