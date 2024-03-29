import numpy as np
import pandas as pd
import re

mail_file = pd.read_csv(" ")
# wipe off all null values
mail_file = mail_file[['Id', 'ExtractedBodyText']].dropna()

#  pre-processing mail data
def clean_email_text(text):
    text = text.replace('\n', " ")
    text = re.sub(r"-", " ", text)  # divide the words which has "-"
    text = re.sub(r"\d + /\d + /\d +", " ", text)  # date info is irrelevant
    text = re.sub(r"[0-2]?[0-9]:[0-6]:[0-9]]", " ", text)  # meaningless time data
    text = re.sub(r"[\w]+@[\.\w] +", " ", text)  # the email addresses are useless
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_] + \. + [A-Za-z0-9\.\/%&=\?\-_]+/i", " ", text)  # Http info is useless
    pure_text = ''
    # in case other special chars, we loop it again nad filter the data
    for letter in text:
        # only leave chars and space
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    # reduce the other meaningless words and get the useful words
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text

# get the document values
docs = mail_file['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))
doclist = docs.values

# get the stop words list
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'most', 'further', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'I', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]

# build the dictionary
dictionary = corpora.Dictionary(texts)
corpora = [dictionary.doc2bow(text) for text in texts]

# build the LDA model
LDA = gensim.models.ldamodel.LdaModel(corpus=corpora, id2word=dictionary, num_topics=20)
LDA.print_topics(num_topics=20, num_words=5)

LDA.get_document_topics(bow=20)
LDA.get_term_topics(word_id=20)

