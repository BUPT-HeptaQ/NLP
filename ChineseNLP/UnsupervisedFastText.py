import jieba
import pandas as pd
import random
import fasttext

category_dictionary = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}

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
entertainment = data_file_entertainment.content.values.tolist()[2000:22000]
military = data_file_military.content.values.tolist()[2000:22000]
sports = data_file_sports.content.values.tolist()[:20000]

# load the stopwords
stopwords = pd.read_csv("D:stopwords.txt", index_col=False, quoting=3,
                        sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values


# preprocessing data
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segments = jieba.lcut(line)
            segments = filter(lambda x: len(x) > 1, segments)
            segments = filter(lambda x: x not in stopwords, segments)
            sentences.append(" ".join(segments), category)

        except (OSError, TypeError) as reason:
            print("the error info is:", str(reason))
            print(line)
            continue


# generate unsupervised train data
sentences = []

preprocess_text(technology, sentences, category_dictionary['technology'])
preprocess_text(car, sentences, category_dictionary['car'])
preprocess_text(entertainment, sentences, category_dictionary['entertainment'])
preprocess_text(military, sentences, category_dictionary['military'])
preprocess_text(sports, sentences, category_dictionary['sports'])

# Shuffle the order to produce a more reliable training set
random.shuffle(sentences)

print("writing data to fastText unsupervised learning format...")
out = open('unsupervised_train_data.txt', 'w')

for sentence in sentences:
    out.write(sentence.encode('utf8') + "\n")
print("done!")

# Skipgram model
model = fasttext.skipgram('unsupervised_train_data.txt', 'model')
print(model.words)  # list of words in dictionary

# CBOW model
model = fasttext.cbow('unsupervised_train_data.txt', 'model')
print(model.words)  # list of words in dictionary

print(model['赛季'])

