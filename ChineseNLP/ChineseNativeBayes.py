# use naive bayes to build a chinese text classifier

""" prepare data """
import jieba
import pandas as pd
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.svm import SVC

data_file_technology = pd.read_csv("D:technology_news.csv", encoding='utf-8')
data_file_technology = data_file_technology.dropna()

data_file_car = pd.read_csv("D:car_news.csv", encoding='utf-8')
data_file_car = data_file_car.dropna()

data_file_entertainment = pd.read_csv("D:entertainment_news.csv", encoding='utf-8')
data_file_entertainment = data_file_entertainment.dropna()

data_file_military = pd.read_csv("D:military_news.csv", encoding='utf-8')
data_file_military = data_file_military.dropna()

data_file_sports = pd.read_csv("D:sports_news.csv", encoding='utf-8')
data_file_sports = data_file_sports.dropna()

technology = data_file_technology.content.values.tolist()[1000:21000]
car = data_file_car.content.values.tolist()[1000:21000]
entertainment = data_file_entertainment.content.values.tolist()[:20000]
military = data_file_military.content.values.tolist()[:20000]
sports = data_file_sports.content.values.tolist()[:20000]

""" split data and chinese text process """
# load the stopwords
stopwords = pd.read_csv("D:stopwords.txt", index_col=False, quoting=3,
                        sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values


def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segments = jieba.lcut(line)
            segments = filter(lambda x: len(x) > 1, segments)
            segments = filter(lambda x: x not in stopwords, segments)
            sentences.append((" ".join(segments), category))

        except (OSError, TypeError) as reason:
            print("the error info is:", str(reason))
            print(line)
            continue


""" generate data set """
sentences = []

preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')

# Shuffle the order to produce a more reliable training set
random.shuffle(sentences)

for sentence in sentences:
    print(sentence[0], sentence[1])

# Divide the original data set into test sets of training sets, using sklearn's own segmentation function "Zip!"
content, tag = zip(*sentences)
content_train, content_test, tag_train, tag_test = train_test_split(content, tag, random_state=1234)
print(len(content_train))

# To extract useful features from noise reduction data, we extract bag of words model features from the text
vectorizer = CountVectorizer(analyzer='word',  # tokenize by character ngrams
                             max_features=4000)  # keep the most common 1000 ngrams
vectorizer.fit(content_train)


def get_features(content):
    vectorizer.transform(content)


# import classifier and train data
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(content_train), tag_train)

""" cross verification part """


# A more reliable method of cross verification is StratifiedKFold,
# but cross verification is the best way to ensure that each sample category is relatively balanced
def stratified_k_fold(content, tag, classifier_class, shuffle=True, n_splits=5, **kwargs):
    sk_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    tag_prediction = tag[:]
    for train_index, test_index in sk_fold.split(content, tag):
        content_train, content_test = content[train_index], content[test_index]
        tag_train = tag[train_index]
        classifier = classifier_class(**kwargs)
        classifier.fit(content_train, tag_train)
        tag_prediction[test_index] = classifier.predict(content_test)

    return tag_prediction


nb = MultinomialNB
print(precision_score(tag, stratified_k_fold(vectorizer.transform(content), np.array(tag), nb), average='macro'))

""" packaging as a class """


class TextClassifier:

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), max_features=20000)

    def features(self, content):
        return self.vectorizer.transform(content)

    def fit(self, content, tag):
        self.vectorizer.fit(content)
        self.classifier.fit(self.features(content), tag)

    def predict(self, content):
        return self.classifier.predict(self.features([content]))

    def score(self, content, tag):
        return self.classifier.score(self.features(content), tag)


text_classifier = TextClassifier()
text_classifier.fit(content_train, tag_train)
print(text_classifier.predict('这 是 有史以来 '))  # here acn input anything
print(text_classifier.score(content_test, tag_test))

# use RBF core
svm = SVC()
svm.fit(vectorizer.transform(content_train), tag_train)
svm.score(vectorizer.transform(content_test), tag_test)

