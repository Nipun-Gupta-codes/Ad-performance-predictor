This project is a simple Streamlit app that rewrites low-performing ad texts into high-performing ad copies using a trained LSTM-based Seq2Seq model.

 Features
- Rewrite ad copies using LSTM encoder-decoder architecture.
- Streamlit UI for easy interaction.
- Tokenizer and model files loaded efficiently using caching.
- Handles padded sequences and outputs generated sequences.

 How It Works
- User enters an ad copy.
- The input is tokenized and padded.
- It’s passed through the encoder to get internal state vectors.
- The decoder uses these states to generate the rewritten version token by token.
- Output is shown on the screen.
