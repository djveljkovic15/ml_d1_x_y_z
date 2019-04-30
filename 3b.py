import statistics

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, nb_features, nb_classes, data, k, weighted=False):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.data = data
        self.k = k
        self.weight = weighted

        # Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
        self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
        self.Y = tf.placeholder(shape=None, dtype=tf.int32)
        self.Q = tf.placeholder(shape=nb_features, dtype=tf.float32)

        # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
        dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)),
                                      axis=1))
        _, idxs = tf.nn.top_k(-dists, self.k)

        self.classes = tf.gather(self.Y, idxs)
        self.dists = tf.gather(dists, idxs)

        if weighted:
            self.w = 1 / self.dists  # Paziti na deljenje sa nulom.
        else:
            self.w = tf.fill([k], 1 / k)

        # Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
        w_col = tf.reshape(self.w, (k, 1))
        self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
        self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)

        # Klasa sa najvise glasova je hipoteza.
        self.hyp = tf.argmax(self.scores)

    # Ako imamo odgovore za upit racunamo i accuracy.
    def predict(self, query_data):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            nb_queries = query_data['x'].shape[0]

            # Pokretanje na svih 10000 primera bi trajalo predugo,
            # pa pokrecemo samo prvih 100.
            # nb_queries = 100
            matches = 0
            for i in range(nb_queries):
                hyp_val = sess.run(self.hyp, feed_dict={self.X: self.data['x'],
                                                        self.Y: self.data['y'],
                                                        self.Q: query_data['x'][i]})
                if query_data['y'] is not None:
                    actual = query_data['y'][i]
                    match = (hyp_val == actual)
                    if match:
                        matches += 1
                    if i % 7 == 0:
                        print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'
                              .format(i + 1, nb_queries, hyp_val, actual, match))
            accuracy = matches / nb_queries
            print('{} matches out of {} examples'.format(matches, nb_queries))
            return accuracy


# Učitavanje i obrada podataka
iris = pd.read_csv('data/iris.csv')

feature_columns = ['sepal_length', 'sepal_width']

x = iris[feature_columns].values
y = iris['species'].values

le = LabelEncoder()
y = le.fit_transform(y)

nacin = 1  # Postoje 2 nacina, prvi je "obican", drugi je sa normalizacijom i nasumicnim mesanjem.
normalizacija = True  # Ukoliko hoces sa normalizacijom (samo drugi nacin)

if nacin == 1:
    # Prvi nacin.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 5, random_state=0)

    nb_train = len(y_train)
    nb_test = len(y_test)

    train_x = np.reshape(x_train, [nb_train, -1])
    test_x = np.reshape(x_test, [nb_test, -1])

    nb_features = 2
    nb_classes = 3
    accuracys = []
    for k in range(1, 16):
        print(k)
        train_data = {'x': x_train, 'y': y_train}
        knn = KNN(nb_features, nb_classes, train_data, k, weighted=False)
        accuracy = knn.predict({'x': x_test, 'y': y_test})
        accuracys.append(accuracy)
        # print('Test set accuracy: ', accuracy)
        print('Test set accuracy za k=' + str(k) + ': ' + str(round(accuracy * 100, 2)) + ' %.')
else:
    # Drugi nacin, sa nasumicnim mesanje, i mogucom normalizacijom.
    nb_features = 2
    nb_classes = 3
    accuracys = []

    data = dict()
    data['x'] = x
    data['y'] = y

    # Nasumično mešanje.
    nb_samples = data['x'].shape[0]
    indices = np.random.permutation(nb_samples)
    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

    if normalizacija is True:
        # Normalizacija,
        data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)

    # Trening-Test delovi.
    training_ratio = 0.8
    test_ratio = 0.2

    nb_train = int(training_ratio * nb_samples)
    data_train = dict()
    data_train['x'] = data['x'][:nb_train]
    data_train['y'] = data['y'][:nb_train]

    nb_test = nb_samples - nb_train
    data_test = dict()
    data_test['x'] = data['x'][nb_train:]
    data_test['y'] = data['y'][nb_train:]

    # Pokrecemo kNN na test skupu.
    for k in range(1, 16):
        print(k)
        # k = 3
        train_data = {'x': data_train['x'], 'y': data_train['y']}
        knn = KNN(nb_features, nb_classes, train_data, k, weighted=False)
        accuracy = knn.predict({'x': data_test['x'], 'y': data_test['y']})
        accuracys.append(accuracy)
        # print('Test set accuracy: ', accuracy)
        print('Test set accuracy za k=' + str(k) + ': ' + str(round(accuracy * 100, 2)) + ' %.')

print('Test set accuracy za sve K zajedno je: ' + str(round(statistics.mean(accuracys) * 100, 2)) + ' %.')

# Crtanje grafa.
for i in range(1, 15):
    plt.plot([i, i + 1], [accuracys[i - 1], accuracys[i]], '-o')

plt.xlabel('K')
plt.ylabel('ACCURACY')
plt.show()

# Komentar:
# U slucaju kada su podaci randomizovani najbolje K varira, obicno pocinje sa manjom accuracy pa raste oko 2-4,
# ali testovi nisu pouzdani.
# U slucaju kada su podaci fiksni imamo najvisi accuracy dok je K=4, a onda naglo opada.
