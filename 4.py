import numpy as np
import pandas as pd
import glob
import os
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import re


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount

    def fit(self, X, Y):
        nb_examples = X.shape[0]

        # Racunamo P(Klasa) - priors
        # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
        # broja u intervalu [0, maksimalni broj u listi]
        self.priors = np.bincount(Y) / nb_examples
        print('Priors:')
        print(self.priors)

        # Racunamo broj pojavljivanja svake reci u svakoj klasi
        occs = np.zeros((self.nb_classes, self.nb_words))
        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w] += cnt
        print('Occurences:')
        print(occs)

        # Racunamo P(Rec_i|Klasa) - likelihoods
        self.like = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.pseudocount
                down = np.sum(occs[c]) + self.nb_words * self.pseudocount
                self.like[c][w] = up / down
        print('Likelihoods:')
        print(self.like)

    def predict(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasu
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = np.log(self.priors[c])
            for w in range(self.nb_words):
                cnt = bow[w]
                prob += cnt * np.log(self.like[c][w])
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        print('\"Probabilites\" for a test BoW (with log):')
        print(probs)
        prediction = np.argmax(probs)
        return prediction

    def predict_multiply(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasu
        # Mnozimo i stepenujemo kako bismo uporedili rezultate sa slajdovima
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = self.priors[c]
            for w in range(self.nb_words):
                cnt = bow[w]
                prob *= self.like[c][w] ** cnt
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        print('\"Probabilities\" for a test BoW (without log):')
        print(probs)
        prediction = np.argmax(probs)
        return prediction


# path = r'data/imdb/'
# tip = 'neg'
# all_files = glob.glob(os.path.join(path + tip, "*.txt"))
# df = pd.concat((pd.read_csv(f) for f in all_files))
#
# print(df)

pos_corpus = []
neg_corpus = []
for root, dirs, files in os.walk("data/imdb", topdown=False):
    for name in files:
        if root.endswith('pos'):
            if name.endswith('.txt'):
                with open(root + '/' + name, 'r', encoding='utf8') as fi:
                    pos_corpus.append(fi.read())
        if root.endswith('pos'):
            if name.endswith('.txt'):
                with open(root + '/' + name, 'r', encoding='utf8') as fi:
                    neg_corpus.append(fi.read())

# print(len(neg_corpus))
# print(len(pos_corpus))

porter = PorterStemmer()
# Cistimo korpus
print('Cleaning the positive corpus...')
clean_pos_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))
for doc in pos_corpus:
    words = wordpunct_tokenize(doc)
    # Remove special chars and convert to lowercase
    words_lower = [w.lower() for w in words]
    # print(words_lower)
    words_remspecial = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_lower]
    words_filtered = [w for w in words_remspecial if w not in stop_punc if w != 'br' if w != '']
    # words_filtered = [w for w in words_lower if w not in stop_punc]
    # print(words_filtered)
    words_stemmed = [porter.stem(w) for w in words_filtered]
    # print('Final:', words_stemmed)
    clean_pos_corpus.append(words_stemmed)
# print(len(clean_pos_corpus))


print('Cleaning the negative corpus...')
clean_neg_corpus = []
for doc in neg_corpus:
    words = wordpunct_tokenize(doc)
    # Remove special chars and convert to lowercase
    words_lower = [w.lower() for w in words]
    # print(words_lower)
    words_remspecial = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_lower]
    words_filtered = [w for w in words_remspecial if w not in stop_punc if w != 'br' if w != '']
    # words_filtered = [w for w in words_lower if w not in stop_punc]
    # print(words_filtered)
    words_stemmed = [porter.stem(w) for w in words_filtered]
    # print('Final:', words_stemmed)
    clean_neg_corpus.append(words_stemmed)
# print(len(clean_neg_corpus))

clean_corpus = np.concatenate((clean_pos_corpus, clean_neg_corpus), axis=None)
# Ostavio mi je '' ?

# print(len(clean_corpus))
# print(clean_corpus)


# 1: Bag of Words model sa 3 različita scoringa
np.set_printoptions(precision=2, linewidth=200)


def occ_score(word, doc):
    return 1 if word in doc else 0


def numocc_score(word, doc):
    return doc.count(word)


def freq_score(word, doc):
    return doc.count(word) / len(doc)


print('Creating BOW features...')
# for score_fn in [occ_score, numocc_score, freq_score]:
#     X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
#     for doc_idx in range(len(clean_corpus)):
#         doc = clean_corpus[doc_idx]
#         for word_idx in range(len(vocab)):
#             word = vocab[word_idx]
#             cnt = score_fn(word, doc)
#             X[doc_idx][word_idx] = cnt
#     print('X:')
#     print(X)
#     print()
nb_words = 10000

f = FreqDist()
freq = FreqDist([w for tw in clean_corpus for w in tw])
best_words, _ = zip(*freq.most_common(nb_words))
clean_corpus_bows = []
for tw in clean_corpus:
    bow = dict()
    for i in range(nb_words):
        cnt = tw.count(best_words[i])
        if cnt > 0:
            bow[i] = cnt
    clean_corpus_bows.append(bow)
print(best_words)
# print(clean_corpus_bows)

# Kreiramo vokabular
print('Creating the vocab...')
vocab_set = set()
for doc in best_words:
    for word in doc:
        vocab_set.add(word)
vocab = list(vocab_set)
# print(len(vocab))

# print('Vocab:', list(zip(vocab, range(len(vocab)))))
print('Feature vector size: ', len(vocab))


for score_fn in [occ_score, numocc_score, freq_score]:
    X = np.zeros((len(best_words), len(vocab)), dtype=np.float32)
    for doc_idx in range(len(best_words)):
        doc = best_words[doc_idx]
        for word_idx in range(len(vocab)):
            word = vocab[word_idx]
            cnt = score_fn(word, doc)
            X[doc_idx][word_idx] = cnt
    print('X:')
    print(X)
    print()










# for name in dirs:
# print(os.path.join(root, name))
#
# # Klase: (china, japan)
# # Vocab: (Chinese, Beijing, Shanghai, Macao, Tokyo, Japan)
# class_names = ['China', 'Japan']
# Y = np.asarray([0, 0, 0, 1])
# X = np.asarray([
#     [2, 1, 0, 0, 0, 0],
#     [2, 0, 1, 0, 0, 0],
#     [1, 0, 0, 1, 0, 0],
#     [1, 0, 0, 0, 1, 1]
# ])
# test_bow = np.asarray([3, 0, 0, 0, 1, 1])
#
# model = MultinomialNaiveBayes(nb_classes=2, nb_words=6, pseudocount=1)
# model.fit(X, Y)
# prediction = model.predict(test_bow)
# print('Predicted class (with log): ', class_names[prediction])
# prediction = model.predict_multiply(test_bow)
# print('Predicted class (without log): ', class_names[prediction])

