from urllib.request import urlopen
from bs4 import BeautifulSoup

import re
import string
import operator

def cleanInput(input):
    input = re.sub('\n+', " ", input).lower()
    input = re.sub('\[[0-9]*\]', " ", input)
    input = re.sub(' +', " ", input)
    input = bytes(input, "UTF-8")
    input = input.decode("ascii", "ignore")
    cleanInput = []
    input = input.split(' ')
    for item in input:
        item = item.strip(string.punctuation)
        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
            cleanInput.append(item)
    return cleanInput

def ngrams(input, n):
    input = cleanInput(input)
    output = {}
    for i in range(len(input)-n+1):
        ngramsTemp = " ".join(input[i:i+n])
        if ngramsTemp not in output:
            output[ngramsTemp] = 0
        output[ngramsTemp] += 1
    return output

def isCommon(ngram):
    commonWords = ["the", "be", "and", "of", "a", "in", "to", "have", "it", "i", "that", "for", "you", "he", "with",
                   "on", "do", "say", "this", "they", "is", "an", "at", "but", "we", "his", "from", "that", "not",
                   "by", "she", "or", "as", "what", "go", "their", "can", "who", "get", "if", "would", "her", "all",
                   "my", "make", "about", "know", "will", "think", "when", "which", "them", "some", "me", "people",
                   "take", "out", "into", "just", "see", "him", "your", "could", "come", "now", "than", "like",
                   "other","how", "then", "its", "our", "two", "more", "these", "want", "way", "look", "first", "also",
                   "new", "because", "day", "use", "no","man", "find", "here", "thing", "give", "many", "well"]
    for word in ngram:
        if word in commonWords:
            return True
    return False

content = str (urlopen("http://pythonscraping.com/files/inaugurationSpeech.txt").read(), 'utf-8')
##bsObj = BeautifulSoup(html)
##content = bsObj.find("div", {"id": "mw-content-text"}).get_text()
ngrams = ngrams(content, 2)
sortedNGrams = sorted(ngrams.items(), key=operator.itemgetter(1),reverse=True)
print(sortedNGrams)
print("2-gram count is: " + str(len(ngrams)))
