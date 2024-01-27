def map_words_to_vectors(words, model, max_length=100):
    word_vectors = []

    for i in words:
        word_vectors.append([model.wv[x] for x in i])

    padded_vectors = pad_sequences(word_vectors, maxlen=max_length, padding='post', truncating='post', dtype='float32')

    # Calculate mean along the correct axis
    mean_vectors = np.mean(padded_vectors, axis=1)
    return mean_vectorss