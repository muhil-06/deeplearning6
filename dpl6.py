
# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import matplotlib.pyplot as plt



data = """deep learning is a powerful tool
deep learning models are useful
machine learning is part of artificial intelligence
artificial intelligence is transforming technology
language modeling helps in text prediction
natural language processing uses machine learning
deep learning improves language understanding
neural networks learn complex patterns
stacked recurrent neural networks improve performance"""



tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

total_words = len(tokenizer.word_index) + 1



input_sequences = []

for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)



model = Sequential()

model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(total_words, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()



history = model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)



plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()



def predict_next_word(seed_text, next_words=1):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len-1,
            padding='pre'
        )

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


print(predict_next_word("deep learning", 2))
print(predict_next_word("artificial intelligence", 2))
print(predict_next_word("machine learning", 2))