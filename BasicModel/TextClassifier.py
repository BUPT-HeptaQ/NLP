import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

# delete the repeated words crudely
def make_word_set(words_files):
    words_set = set()
    with open(words_files, 'r') as fp:
        for line in fp.readlines():
            words = line.strip().encode("utf-8")
            if len(words) > 0 and words not in words_set:
                words_set.add(words)
    return words_set

def text_processing(folder_path, test_size=0.2):  # test_size to divide set
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # traverse all folders
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        # read files
        files_number = 1
        for file in files:
            if files_number > 100:  # avoid break memory, only sample 100 files
                break
            with open(os.path.join(folder_path, file), 'r') as fp:
                raw = fp.read()
            jieba.enable_parallel(4)  # Parallel processing is 4
            word_cut = jieba.cut(raw, cut_all=False)  # exact mode
            word_list = list(word_cut)  # generator turn to list, every word's format is unicode
            jieba.disable_parallel()  # close the parallel mode
            data_list.append(word_list)  # train set list
            class_list.append(folder.decode('utf-8'))  # genre
            files_number += 1

    # divide train set and test set (also could use sklearn to divide)
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # statistic words frequency in all_words_list
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # key function use words frequency by descending order
    # internal function sorted has to be list
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list = list(zip(*all_words_tuple_list[0]))

    return all_words_list, train_data_list, test_data_list, test_data_list, train_class_list, test_class_list

def words_dict(all_words_list, deleteN, stopwords_set=set() ):
    # choose the feature words
    feature_words = []
    feature_dimension = 1
    for t in range(deleteN, len(all_words_list), 1):
        if feature_dimension > 1000:  # feature_words's dimension is 1000
            break

        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            feature_dimension += 1

        return feature_words

# text features
def text_features(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        # estimate module ------------------
        if flag == 'nltk':
            # nltk feature dict
            features = {word: 1 if word in text_words else 0
                        for word in feature_words}

        elif flag == 'sklearn':
            # sklearn feature list
            features = [1 if word in text_words else 0
                        for word in feature_words]
        else:
            features = []
        # ---------------
        return features
    train_feature_list = [text_features(text, feature_words)
                          for text in train_data_list]
    test_feature_list = [text_features(text, feature_words)
                         for text in test_data_list]
    return train_feature_list, test_feature_list

# classification and output the accuracy
def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    # classify module -----------------------
    if flag == 'nltk':
        # use nltk classifier
       train_nltk_list = zip(train_feature_list, train_class_list)
       test_nltk_list = zip(test_feature_list, train_class_list)
       classifier = nltk.classify.NaiveBayesClassifier.train(train_nltk_list)
       classifier = nltk.classify.accuracy(classifier, test_nltk_list)

    elif flag == 'sklearn':
        # use sklearn classifier
       classifier = MultinomialNB().fit(train_feature_list, train_class_list)
       test_accuracy = classifier.score(test_feature_list, test_class_list)

    else:
        test_accuracy = []

    return test_accuracy

print("start")

# text pre-processing
folder_path = ' '
all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path, test_size=0.2)

# generate stopwords_set
stopwords_files = ' '
stopwords_set = make_word_set(stopwords_files)

# sample and classify the text feature
# flag = 'nltk'
flag = 'sklearn'
deleteNs = range(0, 1000, 20)
test_accuracy_list = []
for deleteN in deleteNs:
    # feature_words = words_dict(all_words_list, deleteN)
    feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words, flag)
    test_accuracy = text_classifier(train_feature_list, test_feature_list, train_data_list, test_class_list, flag)
    test_accuracy_list.append(test_accuracy)

print(test_accuracy_list)

# estimate the outcomes
# plt.figure()
plt.plot(deleteNs, test_accuracy_list)
plt.title('Relationship of deleteNs and test_accuracy')
plt.xlabel('deleteNs')
plt.ylabel('test_accuracy')
plt.show()
# plt.savefig('result.png')

print("finished!")

