import jieba.analyse as analyse
import pandas as pd
data_file = pd.read_csv("D:technology_news.csv", encoding='utf-8')
df = data_file.dropna()

lines = data_file.content.values.tolist()
content = " ".join(str(lines))
outcome = " ".join(analyse.extract_tags(content, topK=20, withWeight=False, allowPOS=()))
print(outcome)
