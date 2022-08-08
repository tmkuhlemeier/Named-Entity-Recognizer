#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# FILE: custom_chunker.py

# Natural Language Toolkit: code_classifier_chunker
# Based on code from
#   http://www.nltk.org/book/ch07.html#code-classifier-chunker
#
# Revisions:
# - Not using "megam" as the machine learning engine.
# - The feature builder is a constructor parameter.
# - Added the method `explain()`, which prints the docstring of the feature builder.
# - Added access to the `show_most_informative_features()` method of the underlying classifier.
#
# Alexis Dimitriadis

import nltk
from nltk.chunk.util import conlltags2tree, tree2conlltags

nltk.download('conll2002')

# If numpy is absent, the nltk fails with a very confusing error.
# We avoid problems by checking directly
try:
    import numpy
except ImportError:
    print("You need to download and install numpy!!!")
    raise

import time ###

class ConsecutiveNPChunker(nltk.ChunkParserI):
    """
    Train a classifier on chunked data in Tree format.
    Arguments for the constructor:

    featuremap   The function that will compute features for each word
        in a sentence. See the NLTK book (and the assignment)
        for the arguments it must accept.

    train_sents  A list of sentences in chunked (Tree) format.
    
    algorithm  (optional). The name of the machine-learning model to use.
    """
    def __init__(self, featuremap, train_sents, algorithm="IIS"): ###
        self._algorithm = algorithm
        self._len_trainset = len(train_sents) ###
        tagged_sents = [[((w,t),c) for (w,t,c) in tree2conlltags(sent)]
                            for sent in train_sents]
        start = time.time() ###
        self.tagger = _ConsecutiveNPChunkTagger(featuremap, tagged_sents, algorithm)
        end = time.time() ###
        self._traintime = end - start ###

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return conlltags2tree(conlltags)

    chunk = parse  # A synonym for the absent-minded
    
    def explain(self):
        """Print the docstring of our feature extraction function"""
        print("Algorithm:", self._algorithm)
        # Print the feature map's help string:
        print(self.tagger._featuremap.__doc__)
        
    def chunker_info(self):
        """
        Return info
        Algorithm and feature docstring
        """
        if self._algorithm == "NaiveBayes" :
            return("NaiveBayes\n" + self.tagger._featuremap.__doc__)
        else :
            return("MaxEnt, " + self._algorithm + " algorithm\n" \
            + self.tagger._featuremap.__doc__)
 
    def show_most_informative_features(self, n=10):
        """Call our classifier's `show_most_informative_features()` function."""
        self.tagger.classifier.show_most_informative_features(n)


class _ConsecutiveNPChunkTagger(nltk.TaggerI):
    """This class is not meant to be
    used directly: Use ConsecutiveNPChunker instead."""

    def __init__(self, featuremap, train_sents, algorithm): ###
        
        self._featuremap = featuremap
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = self._featuremap(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        if algorithm == "NaiveBayes" : ###
            self.classifier = nltk.NaiveBayesClassifier.train(train_set) ###
        else : ###
            self.classifier = nltk.MaxentClassifier.train( 
                train_set, algorithm=algorithm, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = self._featuremap(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

