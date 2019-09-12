import re
from collections import Counter

def get_max_value_v1(text):
    text = text.lower()
    result = re.findall('[]a-zA-Z]', text)
    count = Counter(result)
    count_list = list(count.values())
    max_value = max(count_list)
    max_list = []
    for k, v in count.items():
        if v == max_value:
            max_list.append(k)
    max_list = sorted(max_list)
    return max_list[0]


# basic compile pattern
import re

pattern = re.compile(r'he.*\!')
match = pattern.match('hello,zhangbocheng! how are you?')

if match:
    print(match.group())


import jieba
import jieba.analyse as analyse

seg_list = jieba.cut("Harry Potter is a series of fantasy novels written by British author J. K. Rowling.",
                     cut_all=True)
print(seg_list)
print("Full Mode: " + "/ ".join(seg_list))

seg_list = jieba.cut("Harry Potter is a series of fantasy novels written by British author J. K. Rowling.",
                     cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))

seg_list = jieba.cut("The novels chronicle the lives of a young wizard, "
                     "Harry Potter, and his friends Hermione Granger and Ron Weasley")
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("The success of the books and films has allowed the Harry Potter franchise to expand")
print(", ".join(seg_list))

lines = open('Harry Potter.txt').read()
print(" ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())))

