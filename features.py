#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Features
Defines feature extractor functions which can be imported for building 
chunkers.
"""

import re
import csv
from nltk import FreqDist


# open and read needed files :

conn = open("VNC2013.csv", encoding="utf-8")
myreader = csv.reader(conn)
raw_data = [ row for row in myreader ]
conn.close()

conn2 = open("Locations.txt", encoding = "utf-8") # locations adapted from Wikipedia
alltext = conn2.read()
conn2.close()

# make name sets :

# set of the 1000 most common names Dutch
nfdist = FreqDist()
for row in raw_data :
    nfdist[row[0]] = int(row[2])  # row 2 indicates the frequency
nameset = set([n for (n, f) in nfdist.most_common(1000)]) # convert to set

# set of locations of the world (countries and capitals)
locationset = set(alltext.split())
locset = locationset#|blocset


# feature definitions :

def features_simple_1(sentence, i, history):
    """Simplest chunker features: Just the POS tag of the word""" 
    word, pos = sentence[i]
    return { "pos": pos }


def features_small(sentence, i, history):
    """(features_small) 14 features"""
    # kept simple, special for Naive Bayes
    # (most) features are self explanatory
    word, pos = sentence[i]
    capcount = 0
    for letter in word :
        if re.search(r"[A-Z]", letter):
            capcount += 1
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        previob = "O"
    else:
        prevword, prevpos = sentence[i-1]
        previob = history[i-1]
    if i == len(sentence)-1:
        nextpos = "<END>"
    else:
        nextpos = sentence[i+1][1]
    return{ "pos" : pos,
            "capcount" : capcount,
            "word": word,
            "wordlower": word.lower(),
            "prevwordlower": prevword.lower(),
            "prevpos": prevpos,
            "nextpos": nextpos, 
            "upper?": word.isupper(),
            "previob": previob,
            "name?": word in nameset,
            "location?": word in locset,
            "prevname?": prevword in nameset,
            "nationality?" : (word.istitle() and (word.endswith("se") or 
                                           word.endswith("sche"))),
            "wcombination?": (word[0].isupper() and "-" in word)
            }
    

def features_medium(sentence, i, history):
    """(features_medium) 20 features"""
    # more complex
    word, pos = sentence[i]
    capcount = 0
    for letter in word :
        if re.search(r"[A-Z]", letter):
            capcount += 1
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        previob = "O"
    else:
        prevword, prevpos = sentence[i-1]
        previob = history[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return{ "pos" : pos,
            "startcap?" : bool(re.search(r"\b[A-Z]", word)), # start with uppercase
            "capcount" : capcount, # most organizations have 2-4 uppercased letters
            "word": word,
            "prevword": prevword.lower(),
            "prevpos": prevpos,
            "nextpos": nextpos, 
            "cap>1": bool(capcount > 1), # this increased the performance
            "previob": previob,
            'nextword_istitle?': nextword.istitle(),
            'prevword_isupper?': prevword.isupper(),
            "word_isupper?": word.isupper(),
            'word_istitle?': word.istitle(),
            'nextword_isupper?': nextword.isupper(),
            'prevword_istitle?': prevword.istitle(),
            "name?": word in nameset,
            "prevname?": prevword in nameset, # if prevword in nameset 
            # (surnames), it is likely that the current word is a name as well
            "location?": word in locset,
            # nationalities in Dutch start with capitalized letter and end 
            # with "se" or "sche"
            "nationality?" : (word.istitle() and (word.endswith("se") or
                                           word.endswith("sche"))),
            "wcombination?": (word[0].isupper() and "-" in word) # MISC words 
            # often are titled and contain a "-" simultaneously
            }
    
            
def features_complex(sentence, i, history): 
    """(features_complex) 23 features"""
    # complex and large features
    word, pos = sentence[i]
    capcount = 0
    for letter in word :
        if re.search(r"[A-Z]", letter):
            capcount += 1
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        previob = "O"
        prevpreviob = "O"
    else:
        prevword, prevpos = sentence[i-1]
        previob = history[i-1]
    if i <= 1 :
        prevpreviob = "O"
    else :
        prevpreviob = history[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return{ "pos" : pos,
            "startcap?" : bool(re.search(r"\b[A-Z]", word)),
            "capcount" : capcount,
            "wordl": word.lower(),
            "prevword": prevword.lower(),
            "prevpos": prevpos,
            "nextpos": nextpos, 
            "prevpos+pos": "%s+%s" % (prevpos, pos), # pos sequence 
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "cap>1": bool(capcount > 1),
            "previob": previob,
            'nextword_istitle?': nextword.istitle(),
            'prevword_isupper?': prevword.isupper(),
            "word.isupper()": word.isupper(),
            'word.istitle()': word.istitle(),
            'nextword_isupper?': nextword.isupper(),
            'prevword_istitle?': prevword.istitle(),
            "name?": word in nameset,
            "prevname?": prevword in nameset,
            "prevpreviob": prevpreviob,
            "nationality?" : (word.istitle() and (word.endswith("se") or
                                           word.endswith("sche"))),
            "location?": word in locset,
            "wcombination?": (word[0].isupper() and "-" in word)
            }
        