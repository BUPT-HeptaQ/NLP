import warnings
import jieba    # used to split sentences
import numpy
import codecs   # codecs provides open funtion to assign the encode language of file,
                # and turning them into inner unicode when it is loading
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline
import matplotlib
from wordcloud import WordCloud
warnings.filterwarnings("ignore")
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

""" load the entertainment new data and split them"""

data_file = pd.read_csv("D:entertainment_news.csv", encoding='utf-8')
data_file = data_file.dropna()
data_file.head()

content = data_file.content.values.tolist()

# jieba.load_userdict(u"data/user_dic.txt")
segment = []

for line in content:
    try:
        segments = jieba.lcut(line)
        for seg in segments:
            if len(seg) > 1 and seg != '\r\n':
                segment.append(seg)

    except (OSError, TypeError) as reason:
        print("the error info is:", str(reason))
        print(line)
        continue

""" delete the stop words """

words_data_file = pd.DataFrame({'segment': segment})
# print(words_df.head())
stopwords = pd.read_csv("D:stopwords.txt", index_col=False, quoting=3,
                        sep="\t", names=['stopword'], encoding='utf-8')  # quoting=3 noting is quoted
# print(stopwords.head())
words_data_file = words_data_file[~words_data_file.segment.isin(stopwords.stopword)]

""" words frequency statistics"""
words_statistics = words_data_file.groupby(by=['segment'])['segment'].agg({"count": numpy.size})
words_statistics = words_statistics.reset_index().sort_values(by=["count"], ascending=False)
print(words_statistics.head())

word_cloud = WordCloud(font_path="D:simhei.ttf", background_color="white", max_font_size=75)  # font_path = font file
word_frequency = {x[0]: x[1] for x in words_statistics.head(1000).values}
word_cloud = word_cloud.fit_words(word_frequency)
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

# user-defined background to make WordCloud
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
matplotlib.rcParams['figure.figsize'] = (15.0, 15.0)

background_img = imread('D:MAP.jpg')
word_cloud = WordCloud(background_color="white", mask=background_img, font_path='D:simhei.ttf', max_font_size=200)
word_frequency = {x[0]: x[1] for x in words_statistics.head(1000).values}
word_cloud = word_cloud.fit_words(word_frequency)
background_img_colors = ImageColorGenerator(background_img)

plt.imshow(word_cloud.recolor(color_func=background_img_colors))
plt.axis("off")
plt.show()

