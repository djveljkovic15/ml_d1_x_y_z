# TODO popuniti kodom za problem 2a
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


tf.reset_default_graph()
np.set_printoptions(suppress=True, precision=5)
filename = 'data/funky.csv'
all_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0, 1))
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]

nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

nb_features = 3
lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
data_help = dict()
stepeni = dict()

# for lb in lambdas:
for l in range(0, 7):
    for i in range(1, nb_features + 1):
        data_help['x'] = data['x']
        data_help['y'] = data['y']
        print("Krug {}".format(i))
        print('Originalne vrednosti (prve 3):')
        print(data_help['x'][:3])
        print('Feature matrica (prva 3 reda):')
        data_help['x'] = create_feature_matrix(data_help['x'], i)
        print(data_help['x'][:3, :])

        plt.scatter(data_help['x'][:, 0], data_help['y'])
        plt.xlabel('X')
        plt.ylabel('Y')

        X = tf.placeholder(shape=(None, i), dtype=tf.float32)
        Y = tf.placeholder(shape=None, dtype=tf.float32)
        w = tf.Variable(tf.zeros(i))
        bias = tf.Variable(0.0)

        w_col = tf.reshape(w, (i, 1))
        hyp = tf.add(tf.matmul(X, w_col), bias)

        Y_col = tf.reshape(Y, (-1, 1))
        # loss = tf.reduce_mean(tf.square(hyp - Y_col))
        reg = tf.scalar_mul(lambdas[l], tf.square(w_col))  # lambdas[l] = lb
        loss = tf.add(tf.reduce_mean(tf.square(hyp - Y_col)), reg)
        opt_op = tf.train.AdamOptimizer().minimize(loss)  # koristiti ovo

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nb_epochs = 100
        for epoch in range(nb_epochs):
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: data_help['x'][sample].reshape((1, i)),
                        Y: data_help['y'][sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss
            epoch_loss /= nb_samples
            if (epoch + 1) % 100 == 0:
                # np.set_printoptions(precision=5)
                print('Epoch: {}/{}| Avg loss: {}'.format(epoch + 1, nb_epochs, epoch_loss))
                # izbacio sam :0.5 jer ima printoptions precision

        w_val = sess.run(w)
        bias_val = sess.run(bias)
        print('w = ', w_val, 'bias = ', bias_val)
        xs = create_feature_matrix(np.linspace(-2, 4, 100), i)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color='g')
        final_loss = sess.run(loss, feed_dict={X: data_help['x'], Y: data_help['y']})  # ???
        print("Final loss: {}".format(final_loss))
        stepeni[l] = final_loss
        # if l is 6:  # tensorboard --logdir=output
        #             # Komanda da se napravi graf. Kod mene iz nekog razloga ne radi i nisam uspeo da popravim.
        #     writer = tf.summary.FileWriter("output", sess.graph)
        #     writer.close()

writer = tf.summary.FileWriter('output')
writer.add_graph(tf.get_default_graph())
writer.flush()

plt.xlim([-2, 4])
plt.ylim([-3, 4])
plt.show()
for p in range(6):
    plt.plot([stepeni[p][0], stepeni[p + 1][0]], [lambdas[p], lambdas[p + 1]], '-bo')

plt.xlabel('LAMBDA')
plt.ylabel('LOSS')
# plt.scatter(lambdas, stepeni.values())
# plt.plot(lambdas, stepeni.values(), color='g')

plt.show()


# TODO Napisati komentar i napraviti preko TensorBuild prikaz grafa












