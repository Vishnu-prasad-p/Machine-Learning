#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:19:56 2018

@author: apple
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
stop_words = set(stopwords.words('english'))

text = """A strike by farmers in Maharashtra continues to affect normal life, despite the State governmentís announcement of an end to the strike last week. The farmers, whose demands include full waiver of farm loans, hikes in the minimum support price for agricultural produce and writing off of pending electricity bills, have been on an indefinite strike since June 1. As the strike nears the end of its first week, prices of essential goods such as milk, fruits and vegetables have risen steeply, causing distress to consumers. Some farmer groups agreed to call off their strike after Chief Minister Devendra Fadnavis promised that his government would waive farm loans of small and marginal farmers worth about ?30,000 crore, increase power subsidies, hike the price for milk procurement, and also set up a State commission to look into the matter of raising the MSP for crops. He also promised that buying agricultural produce below their MSP would soon be made a criminal offence. Other farmer groups, meanwhile, have stuck to their demand for a complete farm loan waiver and continued with their protest. It is notable that the protests have come soon after the Uttar Pradesh government waived farm loans earlier this year, setting off similar demands in other States. Yet, while Maharashtraís farmers have caught the attention of the government, the focus on quick fixes has pushed aside the real structural issues behind the crisis.
At the root of the crisis is the steep fall in the prices of agricultural goods. The price slump, significantly, has come against the backdrop of a good monsoon that led to a bumper crop. The production of tur dal, for instance, increased five-fold from last year to over 20 lakh tonnes in 2016-17. Irrespective of price fluctuations, MSPs are supposed to enable farmers to sell their produce at remunerative prices. But procurement of crops at MSP by the government has traditionally been low for most crops, except a few staples such as rice and wheat. This has forced distressed farmers to sell their produce at much lower prices, adding to their debt burden. Not surprisingly, the whole system of agricultural marketing has led farmers to feel cheated, and it was only a matter of time before they organised themselves to protest. Going forward, any long-term, wide-scale procurement of crops at MSPs looks unlikely; even a one-time full loan waiver is considered unrealistic by the Chief Minister, given the Stateís finances. The possible ban on buying produce below the MSP would just worsen the crisis by making it hard for farmers to sell their produce even at the market price. The only long-term solution is to gradually align crop production with genuine price signals, while moving ahead with reforms to de-risk agriculture, especially by increasing the crop insurance cover. Expediting steps to reform the Agricultural Produce Market Committee system and introduce the model contract farming law would go a long way to free farmers from MSP-driven crop planning.
"""

#print(sent_tokenize(text))
#print(word_tokenize(text))

tokenized = sent_tokenize(text)
for i in tokenized:
     
    # Word tokenizers is used to find the words 
    # and punctuation in a string
    wordsList = nltk.word_tokenize(i)
 
    # removing stop words from wordList
    wordsList = [w for w in wordsList if not w in stop_words] 
 
    #  Using a Tagger. Which is part-of-speech 
    # tagger or POS-tagger. 
    tagged = nltk.pos_tag(wordsList)
 
    print(tagged)
    
s_t=(sent_tokenize(text))
w_t=(word_tokenize(text))

# Used when tokenizing words
sentence_re = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''
    
lemmatizer = nltk.WordNetLemmatizer()
stemmer = PorterStemmer()

#{<NN.*|JJ>*<NN.*>}
grammar = r"""
    NBAR:
        {<NN.*|VB>}  # Nouns and Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""
chunker = nltk.RegexpParser(grammar)

toks = nltk.regexp_tokenize(text, sentence_re)
postoks = nltk.tag.pos_tag(toks)

#print(postoks)

tree = chunker.parse(postoks)

from nltk.corpus import stopwords
stopwords = stopwords.words('english')


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = stemmer.stem(word)
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term

terms = get_terms(tree)

imp=[]
for term in terms:
    for word in term:
        print(word),
        imp.append(word)
#    print
#    postoks.draw()
postoks1 = nltk.tag.pos_tag(imp)
imptree = chunker.parse(postoks1)