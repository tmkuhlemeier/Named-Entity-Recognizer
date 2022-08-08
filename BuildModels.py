#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Models
This module trains and pickles the different classifiers that are developed.
"""

from nltk.corpus import conll2002 as conll
from custom_chunker import ConsecutiveNPChunker
import pickle
import datetime

from features import features_small, features_medium, features_complex


def buildmodel(modelname, featureset, training, algorithm = "IIS") :
    """Trains and pickles a chunk tag model (NER in this case)."""
    # most print statements are for aesthetic and informative purposes for
    # the training process
    current_time = datetime.datetime.now().strftime("%H:%M:%S")   
    print("\n(" + current_time + ")", "Training", modelname + "...")
    
    Recognizer = ConsecutiveNPChunker(featureset, training, algorithm) # train
    output = open(modelname + ".pickle", "wb")
    pickle.dump(Recognizer, output)
    output.close()
    
    traintime = Recognizer._traintime
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    # hours, minutes, seconds
    hms = str(datetime.timedelta(seconds = traintime)).split(":")
    print("(" + current_time +")", "Done")
    #print("%.2f" % traintime, "seconds")
    # print time it took to train the model together with mean over
    # sentences
    print(int(hms[0]), "h,", int(hms[1]), "min,", "%.2f" % float(hms[2]), "s")
    print("%.4f" % float(traintime / Recognizer._len_trainset),
          "s per sentence\n")

if __name__ == "__main__" :

    training = conll.chunked_sents("ned.train")
    
    buildmodel("Small NER", features_small, training, algorithm = "NaiveBayes")
    buildmodel("Medium NER", features_medium, training, algorithm = "IIS")
    buildmodel("Complex NER", features_complex, training, algorithm = "GIS") # best model
    
    


