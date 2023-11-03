import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

from unidecode import unidecode
import os


class negative_sampling(tf.keras.Model):
  def __init__(self, embedding_dim):
    super(negative_sampling, self).__init__()
    self.center_word_embedding = tf.keras.layers.Dense(embedding_dim,
                                      name="center_word")
    self.context_word_embedding = tf.keras.layers.Dense(embedding_dim, name="context_word")
    self.final_layer = tf.keras.layers.Softmax(name="final_layer")

  def call(self, pair):
    target, context = pair
    center_word = self.center_word_embedding(target)
    context_word = self.context_word_embedding(context)
    mat_mul = tf.einsum('be,bce->bc', center_word, context_word)

    output = self.final_layer(mat_mul)

    return output


def create_negative_sampling_datapoints(words, vocab):

    x_train = []
    y_train = []
    contexts = []

    for i in range(len(words)):
        start_until = max(i-3, 0)
        word_choice = words[:start_until]
        start = min(len(words), i+3)
        word_choice.extend(words[start:])

        if i - 1 >= 0:
            if i-2 >= 0:
                x_train.append(vocab.index(words[i]))
                context, index = create_context(words=vocab, i=words[i-2], word_choice=word_choice)
                contexts.append(context)
                y_train.append(index)
            x_train.append(vocab.index(words[i]))
            context, index = create_context(words=vocab, i=words[i - 1], word_choice=word_choice)
            contexts.append(context)
            y_train.append(index)
        if i+1 <= len(words)-1:
            x_train.append(vocab.index(words[i]))
            context, index = create_context(words=vocab, i=words[i + 1], word_choice=word_choice)
            contexts.append(context)
            y_train.append(index)
            if i+2 <= len(words)-1:
                x_train.append(vocab.index(words[i]))
                context, index = create_context(words=vocab, i=words[i+2], word_choice=word_choice)
                contexts.append(context)
                y_train.append(index)
    return x_train, contexts, y_train


def create_context(words, i, word_choice):
    context = []
    for j in range(4):
        context.append(words.index(random.choice(word_choice)))
    index = random.randint(0, 4)
    context.insert(index, words.index(i))

    return context, index


def get_text_from_file(path: str):

    words = []
    files = list(os.listdir(path))
    for i in range(3):
        with open(path + "/" + files[i]) as f:
            lines = f.read()
            lines = lines.translate(str.maketrans('', '', r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~""")).strip()
            lines_arr = lines.lower().split(" ")
            words.extend(lines_arr)

    final_word_list = []
    for word in words:
        word = unidecode(word)
        if '"' in word:
            word = word.replace('"', '')
        if "'" in word:
            word = word.replace("'", '')
        word = word.strip()
        if "-" in word:
            final_word_list.extend([i for i in word.split("-")])
        else:
            final_word_list.append(word)

    final_word_list = list(filter(lambda x: x != '', final_word_list))



    unique = list(set(final_word_list))
    indices = []
    for word in final_word_list:
        indices.append(unique.index(word))

    return unique, final_word_list, indices


def get_data_generator(x_train, y_train, vocab_size):
    i = 0
    while i < len(x_train):
        x = np.zeros((vocab_size))
        x[x_train[i]] = 1

        y = np.zeros((vocab_size))
        y[y_train[i]] = 1
        yield x, y
        i += 1


def get_data_generator_ns(x_train, contexts,  y_train, vocab_size):
    i = 0
    while i < len(x_train):
        x = np.zeros((vocab_size))
        x[x_train[i]] = 1

        context = []
        for j in contexts[i]:
            temp = np.zeros((vocab_size))
            temp[j] = 1
            context.append(temp)

        y = np.zeros((5))
        y[y_train[i]] = 1
        yield (x, context), y
        i += 1


def organise_data():
    path_to_file = "../dataset.txt"
    path_to_file = "./harry_potter"
    words, all_words, indices = get_text_from_file(path_to_file)

    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
        all_words,
        vocabulary_size=len(words),
        window_size=2,
        negative_samples=0)

    x_train = []
    y_train = []

    for i, j in positive_skip_grams:
        x_train.append(words.index(i))
        y_train.append(words.index(j))

    return x_train, y_train, len(words), words


def organise_data_ns():
    path_to_file = "./harry_potter"
    # path_to_file = '../dataset.txt'
    words, all_words, indices = get_text_from_file(path_to_file)
    x_train, contexts, y_train = create_negative_sampling_datapoints(all_words, words)
    vocab_size = len(words)

    return x_train, contexts, y_train, vocab_size, words


def my_init(shape, dtype=None):
    return tf.random.normal(shape, stddev=0.1)


def embedding(one_hot_vector, weights):
    idx = np.where(one_hot_vector == 1)
    return weights[idx]


def find_similar_word(center_word: str, words, weights):
    center_word_embedding = weights[words.index(center_word)]
    similar_words = []
    for word in words:
        if word == center_word:
            continue
        context_word_embedding = weights[words.index(word)]
        # distance = np.linalg.norm(center_word_embedding - context_word_embedding)
        distance = np.dot(
            center_word_embedding, context_word_embedding
        )/(np.linalg.norm(center_word_embedding)*np.linalg.norm(context_word_embedding))
        if len(similar_words) < 5:
            similar_words.append({
                "word": word,
                "distance": distance
            })
            similar_words = sorted(similar_words, key=lambda x: x['distance'])
        else:
            if distance < similar_words[4]['distance']:
                similar_words.pop(4)
                similar_words.append({
                    "word": word,
                    "distance": distance
                })
                similar_words = sorted(similar_words, key=lambda x: x['distance'])

    return similar_words

def inference(words, weights, all_words):
    word_one = weights[all_words.index(words[0])]
    word_two = weights[all_words.index(words[1])]
    word_three = weights[all_words.index(words[2])]

    # get_to_distance = np.dot(
    #     word_one, word_two
    # ) / (np.linalg.norm(word_one) * np.linalg.norm(word_two))
    get_to_distance = np.linalg.norm(word_one - word_two)
    similar_words = []

    for word in all_words:
        if word == words[2]:
            continue
        test_word_embedding = weights[all_words.index(word)]
        distance = np.linalg.norm(word_three - test_word_embedding)
        # distance = np.dot(
        #     word_three, test_word_embedding
        # )/(np.linalg.norm(word_three)*np.linalg.norm(test_word_embedding))
        distance_between = abs(get_to_distance-distance)
        if len(similar_words) < 10:
            similar_words.append({
                "word": word,
                "distance": distance_between
            })
            similar_words = sorted(similar_words, key=lambda x: x['distance'])
        else:
            if distance_between < similar_words[9]['distance']:
                similar_words.pop(9)
                similar_words.append({
                    "word": word,
                    "distance": distance_between
                })
                similar_words = sorted(similar_words, key=lambda x: x['distance'])

    return similar_words




def main():
    # Get datapoints

    # Organise data for model without negative sampling
    x_train, y_train, vocab_size, word_list = organise_data()

    # Organise data for model with negative sampling
    x_train, contexts, y_train, vocab_size, word_list = organise_data_ns()

    embedding_dim = 20

    # Dataset for model without negative sampling
    dataset = tf.data.Dataset.from_generator(
        generator=get_data_generator,
        args=[x_train, y_train, vocab_size],
        output_signature=(tf.TensorSpec(shape=(vocab_size), dtype=tf.float32), tf.TensorSpec(shape=(vocab_size), dtype=tf.float32))
        )

    # Dataset for model with negative sampling
    dataset = tf.data.Dataset.from_generator(
        generator=get_data_generator_ns,
        args=[x_train, contexts, y_train, vocab_size],
        output_signature=((tf.TensorSpec(shape=(vocab_size), dtype=tf.float32), tf.TensorSpec(shape=(5, vocab_size), dtype=tf.float32)),  tf.TensorSpec(shape=(5), dtype=tf.float32))
        )

    BATCH_SIZE = 100
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


    # Negative sampling model
    model = negative_sampling(embedding_dim=embedding_dim)

    # Model without negative sampling
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=embedding_dim, kernel_initializer=my_init, input_shape=(vocab_size,)),
        tf.keras.layers.Dense(units=vocab_size, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    )

    history = model.fit(dataset, epochs=30)

    word_embeddings, bias = model.layers[0].get_weights()

    # # Plot embeddings (first reduce to 2 dimensions)
    pca = PCA(n_components=2)
    pca.fit(word_embeddings)
    reduced_embeddings = pca.transform(word_embeddings)

    x_coords = []
    y_coords = []

    # Select words to graph
    # words_to_graph = ["harry", "draco", "ron", "hermione", "malfoy", "weasley", "granger", "potter", "bertie", "botts", "fang", "hagrid", "gryffindor", "ravenclaw", "ravenclaws", "slytherin", "hufflepuff", "quill", "voldemort", "quidditch", "crabbe", "goyle", "snitch", "sorcerers", "stone", "professor", "snape", "dumbledore", "mcgonagall", "beard", "scar", "quirrel", "broom", "invisibility"]

    for i in range(len(word_list)):
        x_coords.append(reduced_embeddings[i][0])
        y_coords.append(reduced_embeddings[i][1])

    plt.scatter(x=x_coords, y=y_coords)
    for i in range(len(word_list)):
        plt.annotate(word_list[i], (x_coords[i], y_coords[i]))

    plt.show()

    inference_words = inference(words=['harry', 'gryffindor', 'draco'], weights=word_embeddings, all_words=word_list)
    for word in inference_words:
        print(word)


    print('-'*10)

    similar_words = find_similar_word(center_word="quidditch", words=word_list, weights=word_embeddings)
    for word in similar_words:
        print(word)


if __name__ == "__main__":
    main()
