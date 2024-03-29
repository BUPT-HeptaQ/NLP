import nltk
# nltk.download('brown')

from nltk.corpus import brown

# pre-processing the dictionary, give the words begin and end symbol
brown_tags_words = []
for sent in brown.tagged_sents():
    brown_tags_words.append(("START", "START"))
    # we abridge tags into two letter
    brown_tags_words.extend([tag[:2], word] for (word, tag) in sent)
    brown_tags_words.append(("END", "END"))


# P(wi|ti) = count(wi,ti)/count(ti)
# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)


# P(ti|t{i-1}) = count(t{i-1}, ti)/count(t{i-1})
brown_tags = [tag for (tag, word) in brown_tags_words]
# count(t{i-1} ti)  bigram means two groups connect together
cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# P(ti|t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

# implement of Viterbi
distinct_tags = set(brown_tags)
# input any content of the sentence(use " ", to divide)
sentence = ["I", "want", "to", "win"]
sentence_len = len(sentence)
# loop the sentence from 1 to N, note as i, every time find the end node as tag X, length of tag link is i
Viterbi = []
# loop the sentence from 1 to N, note as i, label the Tag before every tag X
backpointer = []
first_viterbi = {}
first_backpointer = {}
for tag in distinct_tags:
    # do not record anything for the START tag
    if tag == "START":
        continue
    first_viterbi[tag] = cpd_tags["START"].prob(tag)*cpd_tagwords[tag].prob(sentence[0])
    first_backpointer[tag] = "START"

    print(first_backpointer)
    print(first_viterbi)

# store everything above in Viterbi and Backpointer variables
Viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

# find the best tag
best_tag = max(first_viterbi.keys(), key=lambda tag: first_viterbi[tag])
print("Word", "'" + sentence[0] + "'", "current best two-tag sequence:", first_backpointer[best_tag], best_tag)

for wordindex in range(1, len(sentence)):
    this_viterbi = {}
    this_backpointer = {}
    prev_viterbi = Viterbi[-1]

    for tag in distinct_tags:
        # START has no us, we could ignore it
        if tag == "START":
            continue
        # prev_viterbi[Y]*P(X|Y)*P(w|X)
        best_previous = max(prev_viterbi.keys(), key=lambda prevtag:
                            prev_viterbi[prevtag]*cpd_tags[prevtag].prob(tag)*
                            cpd_tagwords[tag].prob(sentence[wordindex]))
        this_viterbi[tag] = prev_viterbi[best_previous]*cpd_tags[best_previous].prob(tag)*\
                            cpd_tagwords[tag].prob(sentence[wordindex])
        this_backpointer[tag] = best_previous

    # after we find the Y, ever time we should save it
    best_tag = max(this_viterbi.keys(), key=lambda tag: this_viterbi[tag])
    print("Word", "'" + sentence[wordindex] + "'", "current best two-tag sequence:",
          this_backpointer[best_tag], best_tag)

# store everything above in Viterbi and Backpointer variables
Viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

# find every tag sequence which end by END
prev_viterbi = Viterbi[-1]
best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag]*cpd_tags[prevtag].prob("END"))
prob_tagsequence = prev_viterbi[best_previous]*cpd_tags[best_previous].prob("END")
best_tagsequence = ["END", best_previous]
backpointer.reverse()

current_best_tag = best_previous
for best in backpointer:
    best_tagsequence.append(best[current_best_tag])
    current_best_tag = best[current_best_tag]

best_tagsequence.reverse()
print("The sentence was:", end=" ")
for w in sentence:
    print(w, end=" " "\n")
    print("The best tag sequence is:", end=" ")

for t in best_tagsequence:
    print(t, end=" " "\n")
    print("The probability of the best tag sequence is:", prob_tagsequence)

