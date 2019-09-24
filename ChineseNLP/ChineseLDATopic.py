rom gensim import corpora, models, similarities
import gensim
import pandas as pd
import jieba

# load the stopwords
stopwords = pd.read_csv("D:stopwords.txt", index_col=False, quoting=3,
                        sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

# convert to suitable format
data_file = pd.read_csv("D:technology_news.csv", encoding='utf-8')
data_file = data_file.dropna()
lines = data_file.content.values.tolist()

sentences = []
for line in lines:
    try:
        segments = jieba.lcut(line)
        segments = filter(lambda x: len(x) > 1, segments)
        segments = filter(lambda x: x not in stopwords, segments)
        sentences.append(segments)

    except (OSError, TypeError) as reason:
        print("the error info is:", str(reason))
        print(line)
        continue

# bag of words model
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

# LDA model
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
# print No.3 class
print(lda.print_topic(3, topn=5))
#  print all topics
for topic in lda.print_topics(num_topics=20, num_words=8):
    print(topic[1])
    
    
