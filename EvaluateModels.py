#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate Models
Module contains definitions which can be used for evaluating pickled 
models. The main script calls evaluate_model that loads and evaluates 
specified models. 
It prints information about the model and evaluation. 
This module can be imported to be used in other scripts.

"""

from nltk.corpus import conll2002 as conll
import pickle
import datetime


def evaluate_model(pickled_model, testing) :
    """
    Takes a pickled chunk tag model (NER in this case), 
    unpickles (loads) the model and evaluates it on the specified test data. 
    It prints to the console and writes to a file (Evaluation-output.txt) :
    date of evaluation, name, algorithm, feature docstring, the size of the
    training data and how long the training took, chunkparse score and
    evaluation metrics.
    """

    current_time = datetime.datetime.now().strftime("%d-%m-%Y at %H:%M:%S") # dutch time 
    Chunker = pickle.load(open(pickled_model, "rb"))
    fname_parts = pickled_model.split(".")
    
    printwrite("\n" + "(evaluated on " + current_time + ":)")
    printwrite("MODEL:", fname_parts[0])
    printwrite(Chunker.chunker_info())
    printwrite(Chunker._len_trainset, "train sents,",
               len(testing), "test sents")
    traintime = Chunker._traintime
    hms = str(datetime.timedelta(seconds = traintime)).split(":")
    printwrite("Training took", int(hms[0]), "h,", int(hms[1]), "min,",
          "%.2f" % float(hms[2]), "s", end = " ")
    printwrite("(" + ("%.4f" % float(traintime / Chunker._len_trainset)) \
                 + "s per sentence)\n")
    printwrite("Evaluating...", end = " ")
    metrics = Chunker.evaluate(testing) # print accuracy, Pr, Re and F-score
    printwrite("Done")
    printwrite(metrics)
    printwrite("")
    
    correct = metrics.correct()
    guessed = metrics.guessed()
    found = [ v[1] for v in metrics._tp ]
    incorrect = metrics.incorrect()
    
    # adapted from notebook
    printwrite("Actual NEs in the test data:", len(correct))
    printwrite("Chunks guessed (proposed) by the recognizer:", len(guessed))
    printwrite("Found correctly (truepos):", len(found))
    printwrite("Incorrect guesses (falsepos):", len(incorrect))
    printwrite("Missed NEs: (falseneg)", len(metrics.missed()))
    printwrite("")
    #Chunker.show_most_informative_features(30) # only works with NaiveBayes

def printwrite(*args, **kwargs) :
    """Function that prints to the console and to Evaluation-output.txt"""
    
    print(*args, **kwargs)
    with open("Evaluation-output.txt", "a") as outfile :
        print(file = outfile, *args, **kwargs)
        

if __name__ == "__main__" :

    testing = conll.chunked_sents("ned.testa")
    
    evaluate_model("Small NER.pickle", testing)
    evaluate_model("Medium NER.pickle", testing)
    evaluate_model("Complex NER.pickle", testing) # best model (best.pickle)
    #evaluate_model("best.pickle", testing) # (Complex NER) (uncomment to evaluate)
    printwrite("\nBest model: Complex NER (a.k.a. best.pickle)\n")
