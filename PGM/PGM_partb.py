#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:11:27 2018

@author: apple
"""


from nltk.tokenize import sent_tokenize, word_tokenize
 
text = "One day Buddha was walking through a village. A very angry and rude young man came up and began insulting him. “You have no right teaching others,” he shouted. “You are as stupid as everyone else. You are nothing but a fake.”"+\
"Buddha was not upset by these insults. Instead he asked the young man “Tell me, if you buy a gift for someone, and that person does not take it, to whom does the gift belong?”"+\
"The man was surprised to be asked such a strange question and answered, “It would belong to me, because I bought the gift.”"+\
"The Buddha smiled and said, “That is correct. And it is exactly the same with your anger.If you become angry with me and I do not get insulted, then the anger falls back on you."


print(sent_tokenize(text))
print(word_tokenize(text))

