What fastText does is to turn all the words in the document into vectors through lookup table,
take the average and get the classification results directly with the linear classifier.
fastText is similar to the deep averaging network(DAN) on acl-15,
and is a simplified version that removes the hidden layer in the middle.
The paper points out that for some simple classification tasks,
it is not necessary to use too complex network structure to achieve similar results.

Model architecture of fastText for a sentence with N ngram features x1,...,xN.
The features are embedded and averaged to form the hidden variable

There are two tricks mentioned in the fastText paper:

Hierarchical softmax:
With a large number of categories, accelerate the calculation of the softmax layer by building a Huffman code tree,
the same as trick in word2vec before

N - gram features:
Using unigram alone will throw out word order information,
so use hashing to reduce N-gram storage by adding N-gram features

FastText requires the following storage form for text classification:

__label__2 , birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta high school in lakewood township new
jersey. it was founded by rabbi shmuel zalmen stein in 2001 after his father rabbi chaim stein asked him to open a
branch of telshe yeshiva in lakewood . as of the 2009-10 school year the school had an enrollment of 76 students and 6 .
 6 classroom teachers ( on a fte basis ) for a student–teacher ratio of 11 . 5 1 .
__label__6 , motor torpedo boat pt-41 , motor torpedo boat pt-41 was a pt-20-class motor torpedo boat of
the united states navy built by the electric launch company of bayonne new jersey . the boat was laid down as motor boat
 submarine chaser ptc-21 but was reclassified as pt-41 prior to its launch on 8 july 1941 and was completed on 23 july
 1941 .
__label__11 , passiflora picturata , passiflora picturata is a species of passion flower in the passifloraceae family .
__label__13 , naya din nai raat , naya din nai raat is a 1974 bollywood drama film directed by a . bhimsingh . the film
is famous as sanjeev kumar reprised the nine-role epic performance by sivaji ganesan in navarathri ( 1964 )
which was also previously reprised by akkineni nageswara rao in navarathri ( telugu 1966 ) . this film had enhanced his
status and reputation as an actor in hindi cinema .

__label__ is a prefix or we can self-defined, the categories are followed by __label__
