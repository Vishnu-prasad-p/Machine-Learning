#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:37:09 2018

@author: apple
"""

import nltk
from nltk.stem.porter import *
import nlpnet

import spacy
from nltk import Tree
from nltk.tokenize import sent_tokenize, word_tokenize

import en_core_web_sm
import pandas as pd
import re
from collections import OrderedDict
import matplotlib
"""
text = "One day Buddha was walking through a village. A very angry and rude young man came up and began insulting him. “You have no right teaching others,” he shouted. “You are as stupid as everyone else. You are nothing but a fake.”"+\
"Buddha was not upset by these insults. Instead he asked the young man “Tell me, if you buy a gift for someone, and that person does not take it, to whom does the gift belong?”"+\
"The man was surprised to be asked such a strange question and answered, “It would belong to me, because I bought the gift.”"+\
"The Buddha smiled and said, “That is correct. And it is exactly the same with your anger.If you become angry with me and I do not get insulted, then the anger falls back on you."
"""

#text="""Law of attraction is the law of creation. Quantum physicist tells us that the entire universe emerged
#from thought! You create your life through your thoughts and the law of attraction, and every single
#person does the same. It doesn’t just work if you know about it. It has always been working in your life
#and every other person’s life throughout history. When you become aware of this great law, then you
#become aware of how incredibly powerful you are, to be able to think your life into existence."""

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

#Taken from Su Nam Kim Paper...{<NN.*|JJ>*<NN.*>}
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

#%%
class Context_Intent:
    def __init__(self):
        self.question = ""
        self.nlp = en_core_web_sm.load()

    def preprocessing(self, question):
        self.question = question
        split_sent = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\,)(\s|[A-Z].*)', self.question)
        sentences= []
        s=[]
        sents =[x.strip() for x in split_sent if x.strip()]
        #print(sents)
        for i in sents:
            if "and" in i:
                s.append(i)
                s.append(re.split("and",i))
            elif "also" in i:
                s.append(i)
                s.append(re.split("also",i))
            sentences.append(i)

        flat_list = self.flattern(s)
        #print(sentences)

        lis2= [ele for ele in OrderedDict.fromkeys(flat_list) if ele not in set(sentences)]
        lis1= [ele for ele in OrderedDict.fromkeys(sentences) if ele not in set(flat_list)]

        self.new_sentences = lis2+lis1
        #print(new_sent)

        self.context_intent(self.new_sentences)
        return
    def flattern(self,A):     #flatten the list
        rt = []
        for i in A:
            if isinstance(i,list): rt.extend(self.flattern(i))
            else: rt.append(i)
        return rt
    def replace(self,string, substitutions):      #gramma correction (article insertion)
        substrings = sorted(substitutions, key=len, reverse=True)
        regex = re.compile('|'.join(map(re.escape, substrings)))
        return regex.sub(lambda match: substitutions[match.group(0)], string)

    def context_intent(self,new_sentences):
        df2 = pd.DataFrame([])

        for i in self.new_sentences:
            doc = self.nlp(i)
            tokens =[]
            token_head =[]
            child =[]
            dep = []
            dt = {}
            for token in doc:
                tokens.append(token.text)
                token_head.append(token.head.text)
                child.append([child for child in token.children])
                dep.append(token.dep_)
            df1 = pd.DataFrame({"Context1": tokens, "Intent1": token_head, "Dependencies": child, "Dep": dep})


            if df1.empty:
                pass
            else:
                r_intent = df1[df1['Dep'].str.contains("ROOT")]
                r_intnt = r_intent['Intent1'].tolist()
                r_Intent = ''.join(r_intnt)
                context = r_intent['Dependencies'].tolist()
#                dep_context = ''.join(context)


#            print(df1)
            df=df1
#            dt.setdefault(context, r_Intent)
#            print(r_Intent)

obj = Context_Intent()

for sen in s_t:
    inp = sen
    obj.preprocessing(inp)
        
        
        
        