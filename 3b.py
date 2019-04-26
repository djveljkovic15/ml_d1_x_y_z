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


iris = pd.read_csv('data/iris.csv')

feature_columns = ['sepal_length', 'sepal_width']  # , 'petal_length', 'petal_width']

x = iris[feature_columns].values
y = iris['species'].values

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=0)

nb_train = len(y_train)
nb_test = len(y_test)

train_x = np.reshape(x_train, [nb_train, -1])
test_x = np.reshape(x_test, [nb_test, -1])

nb_features = 2  # 4
nb_classes = 3
for k in range(1, 16):
    print(k)
    # k = 3
    train_data = {'x': x_train, 'y': y_train}
    knn = KNN(nb_features, nb_classes, train_data, k, weighted=False)
    accuracy = knn.predict({'x': x_test, 'y': y_test})
    # print('Test set accuracy: ', accuracy)
    print('Test set accuracy za k=' + str(k) + ': ' + str(round(accuracy * 100, 2)) + ' %.')


# TODO Napraviti grafik i napisati komentar
