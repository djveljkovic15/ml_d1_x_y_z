import numpy as np
from random import shuffle
import os
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import re


class MultinomialNaiveBayes:

    def __init__(self, number_of_classes, number_of_words, pseudocount):
        self.number_of_classes = number_of_classes
        self.number_of_words = number_of_words
        self.pseudocount = pseudocount
        self.priors = None
        self.likelihoods = None

    def fit(self, x, y):
        self.priors = np.bincount(y) / len(y)
        self.likelihoods = np.zeros((self.number_of_classes, self.number_of_words))

        word_occurrence = self.calculate_word_occurrences(x, y)

        # calculate likelihoods
        for c in range(self.number_of_classes):
            for w in range(self.number_of_words):
                up = word_occurrence[c][w] + self.pseudocount
                down = np.sum(word_occurrence[c]) + self.number_of_words * self.pseudocount
                self.likelihoods[c][w] = up / down

    def calculate_word_occurrences(self, x, y):
        word_occurrence = np.zeros((self.number_of_classes, self.number_of_words))

        number_of_examples = len(y)

        for i in range(number_of_examples):
            # class
            c = y[i]

            for word, count in x[i].items():
                word_occurrence[c][word] += count

        return word_occurrence

    def predict(self, xs):
        predictions = []

        for i in range(len(xs)):
            log_probabilities = np.zeros(self.number_of_classes)

            # class
            for c in range(self.number_of_classes):
                log_probability = np.log(self.priors[c])

                for word, count in xs[i].items():
                    log_probability += count * np.log(self.likelihoods[c][word])

                log_probabilities[c] = log_probability

            predictions.append(np.argmax(log_probabilities))

        return predictions


class Priprema:
    def __init__(self):
        self.neg_corpus = []
        self.pos_corpus = []
        self.clean_neg_corpus = []
        self.clean_pos_corpus = []
        self.clean_corpus = []
        self.clean_corpus_bows = []
        self.clean_pos_corpus_bows = []
        self.clean_neg_corpus_bows = []
        self.best_words = []
        self.pos_X = []
        self.nb_data = 1250
        self.nb_words = 10000
        self.nb_classes = 2
        self.train_data = int(self.nb_data * 0.8)
        self.test_data = int(self.nb_data * 0.2)
        self.train = None
        self.test = None
        self.neg_X = []
        self.pos_X = []

        self.vocablen = None

        self.ucitaj()
        self.ocisti()

        self.sparse()
        # self.fitur()
        self.test_train_features()
        self.randomize_train()

        np.set_printoptions(precision=2, linewidth=200)

    # Ucitaj korpuse
    def ucitaj(self):
        print("-->Ucitaj")
        for root, dirs, files in os.walk("data/imdb", topdown=False):
            for name in files:
                if root.endswith('pos'):
                    if name.endswith('.txt'):
                        with open(root + '/' + name, 'r', encoding='utf8') as fi:
                            self.pos_corpus.append(fi.read())
                if root.endswith('pos'):
                    if name.endswith('.txt'):
                        with open(root + '/' + name, 'r', encoding='utf8') as fi:
                            self.neg_corpus.append(fi.read())

    # Cistimo korpuse i pravimo jedan zajednicki
    def ocisti(self):
        print("-->Ocisti")
        porter = PorterStemmer()
        stop_punc = set(stopwords.words('english')).union(set(punctuation))
        print('Cleaning the positive corpus...')
        for doc in self.pos_corpus:
            words = wordpunct_tokenize(doc)
            words_lower = [w.lower() for w in words]
            words_remspecial = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_lower]
            words_filtered = [w for w in words_remspecial if w not in stop_punc if w != 'br' if w != '']
            words_stemmed = [porter.stem(w) for w in words_filtered]
            self.clean_pos_corpus.append(words_stemmed)
        print('Cleaning the negative corpus...')
        for doc in self.neg_corpus:
            words = wordpunct_tokenize(doc)
            words_lower = [w.lower() for w in words]
            words_remspecial = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_lower]
            words_filtered = [w for w in words_remspecial if w not in stop_punc if w != 'br' if w != '']
            words_stemmed = [porter.stem(w) for w in words_filtered]
            self.clean_neg_corpus.append(words_stemmed)
        self.clean_corpus = np.concatenate((self.clean_pos_corpus, self.clean_neg_corpus), axis=None)
        # print(len(self.clean_corpus))

    # 1: Bag of Words model sa 3 različita scoringa
    def fitur(self):
        print("-->Fitur")

        def occ_score(word, doc):
            return 1 if word in doc else 0

        def numocc_score(word, doc):
            return doc.count(word)

        def freq_score(word, doc):
            return doc.count(word) / len(doc)

        # Uzmemo najbolje reci
        freq = FreqDist([w for tw in self.clean_corpus for w in tw])
        best_words, _ = zip(*freq.most_common(self.nb_words))

        # Kreiramo vokabular
        print('Creating the vocab...')
        vocab_set = set()
        for doc in best_words:
            for word in doc:
                vocab_set.add(word)
        vocab = list(vocab_set)

        print('Creating BOW features...')
        for score_fn in [occ_score, numocc_score, freq_score]:
            self.pos_X = np.zeros((len(self.clean_pos_corpus), len(vocab)), dtype=np.float32)
            for doc_idx in range(len(self.clean_pos_corpus)):
                doc = self.clean_pos_corpus[doc_idx]
                for word_idx in range(len(vocab)):
                    word = vocab[word_idx]
                    cnt = score_fn(word, doc)
                    self.pos_X[doc_idx][word_idx] = cnt
            print('X:')
            print(self.pos_X)
            print()
        for score_fn in [occ_score, numocc_score, freq_score]:
            self.neg_X = np.zeros((len(self.clean_neg_corpus), len(vocab)), dtype=np.float32)
            for doc_idx in range(len(self.clean_neg_corpus)):
                doc = self.clean_neg_corpus[doc_idx]
                for word_idx in range(len(vocab)):
                    word = vocab[word_idx]
                    cnt = score_fn(word, doc)
                    self.neg_X[doc_idx][word_idx] = cnt
            print('X:')
            print(self.neg_X)
            print()

    def sparse(self):
        print("-->Sparse")
        np.set_printoptions(precision=2, linewidth=200)
        print('Creating BOW features...')
        # Uzmemo najbolje reci
        freq = FreqDist([w for tw in self.clean_corpus for w in tw])
        self.best_words, _ = zip(*freq.most_common(self.nb_words))

        # Sparse representation
        for tw in self.clean_corpus:
            bow = dict()
            for i in range(self.nb_words):
                cnt = tw.count(self.best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            self.clean_corpus_bows.append(bow)
        shuffle(self.clean_corpus_bows)
        print("Creating pos features...")
        for tw in self.clean_pos_corpus:
            bow = dict()
            for i in range(self.nb_words):
                cnt = tw.count(self.best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            self.clean_pos_corpus_bows.append(bow)
        shuffle(self.clean_pos_corpus_bows)
        print("Creating neg features...")
        for tw in self.clean_neg_corpus:
            bow = dict()
            for i in range(self.nb_words):
                cnt = tw.count(self.best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            self.clean_neg_corpus_bows.append(bow)
        shuffle(self.clean_neg_corpus_bows)

        # Kreiramo vokabular
        print('Creating the vocab...')
        vocab_set = set()
        for doc in self.best_words:
            for word in doc:
                vocab_set.add(word)
        vocab = list(vocab_set)
        self.vocablen = len(vocab)

    def test_train_features(self):
        print("-->TT feat")

        # Class labels, 1 for positive, 0 for negative
        positive_labels = np.ones(self.nb_data, dtype=int)
        negative_labels = np.zeros(self.nb_data, dtype=int)

        self.train = {
            'x': (self.clean_pos_corpus_bows[self.test_data:] + self.clean_neg_corpus_bows[self.test_data:]),
            'y': (np.concatenate([positive_labels[self.test_data:], negative_labels[self.test_data:]]))
        }
        print(len(self.train))
        self.test = {
            'x': (self.clean_pos_corpus_bows[:self.test_data] + self.clean_neg_corpus_bows[:self.test_data]),
            'y': (np.concatenate([positive_labels[:self.test_data], negative_labels[:self.test_data]]))
        }
        print(len(self.test))

    def randomize_train(self):
        print("-->Randomize")
        self.train['x'] = np.array(self.train['x'])
        indices = np.random.permutation(len(self.train['x']))
        self.train['x'] = self.train['x'][indices]
        self.train['y'] = self.train['y'][indices]


data = Priprema()
train_data = data.train
test_data = data.test

model = MultinomialNaiveBayes(2, data.vocablen, 1)
print("ovo ide u fit")
print(len(train_data['x']))
print(len(train_data['y']))

model.fit(train_data['x'], train_data['y'])


preds = model.predict(test_data['x'])
nb_correct = 0
nb_total_test = len(preds)
for i in range(nb_total_test):
    if test_data['y'[i]] == preds[i]:
        nb_correct = nb_correct + 1

accuracy = nb_correct / nb_total_test
print('Test set accuracy:' + str(round(accuracy * 100, 2)) + ' %.')

# Ne mogu da pronadjem sta sam promasio, jbg
