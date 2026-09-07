import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# 1. Load Models and Tokenizers
# -------------------------
@st.cache_resource
def load_all():
    encoder_model = tf.keras.models.load_model("model/encoder_model_inf.h5")
    decoder_model = tf.keras.models.load_model("model/decoder_model_inf.h5")
    
    with open("model/input_tokenizer.pkl", "rb") as f:
        input_tokenizer = pickle.load(f)
        
    with open("model/target_tokenizer.pkl", "rb") as f:
        target_tokenizer = pickle.load(f)

    return encoder_model, decoder_model, input_tokenizer, target_tokenizer

encoder_model, decoder_model, input_tokenizer, target_tokenizer = load_all()

# ✅ Define these constants (ensure they match what you used during training)
max_input_len = 50
max_target_len = 50

# -------------------------
# 2. Define Decode Function
# -------------------------
def decode_sequence(input_seq):
    # Encode input
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1), dtype='int32')
    target_seq[0, 0] = target_tokenizer.word_index.get('<start>', 1)

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1), dtype='int32')
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("🚀 Ad Rewriter using LSTM")
st.write("Generate a high-performing rewritten ad using an LSTM-based sequence-to-sequence model.")

user_input = st.text_area("✍️ Enter your ad copy below:")

if st.button("Rewrite"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid ad.")
    else:
        seq = input_tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=max_input_len, padding='post')
        rewritten_ad = decode_sequence(padded_seq)
        st.success("✅ Rewritten Ad:")
        st.write(f"**{rewritten_ad}**")
