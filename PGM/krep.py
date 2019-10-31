#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:30:30 2018

@author: apple
"""

#!/usr/bin/env python3
"""
    knowledge_rep.py - Knowledge representation for each level description
    Author: Dung Le (dungle@bennington.edu)
    Date: 01/10/2017
"""

import nltk
from nltk.corpus import stopwords
import re
import spacy
from rake_nltk import Rake
from nltk.parse.stanford import StanfordDependencyParser

'''
    In level-desc.txt, there are two main components: list of rooms and objects
    and level interactions. For each object in the room, there will be information
    represented in the following format:
    object1(location, relative_position, possible_action, destination, action_requirement)
        location: object current location (which room is the object in?)
        relative_position: where the object is inside the room? (optional)
        possible_action: what is the possible interaction? (can be NULL)
        destination: where will the action lead? Or what is the consequence? (can be NULL)
        action_requirement: special requirement to perform action (can be NULL)
    Example:
    wooden_door(desert, NULL, open, Room-A, NULL)
    ladder(Room-C, NULL, climb up, Room-A, NULL)
'''

stop_words = stopwords.words('english')
stop_words.append('.')
stop_words.append(',')
stop_words.append('!')
POS_tagger = spacy.load('en')

with open("/Users/apple/Desktop/level-desc.txt", "r") as desc_file:
    desc_text = desc_file.read().splitlines()

def description_formatter(description):
    '''
        Format the level description (in text file) into dictionary with key
        being each room and value being a tuple containing objects and
        interactions within the room
        {
            "room-a": "an orange door on the right, a hallway, a ladder going down",
            "room-a-interactions": "you can open the orange door to find room-b",
            ...
        }
    '''
    obj_dict = {}
    desc_dict = {}
    for sent in description:
        if re.search(r'.:.', sent):
            objects = sent.lower().split(sep=':')
            if objects[0] in obj_dict or objects[0] == 'winning':
                desc_dict[objects[0]] = objects[1][1:]
            else:
                obj_dict[objects[0]] = objects[1][1:]
    return obj_dict, desc_dict

def knowledge_representation(object_dict, description_dict):
    knowledge_dict = {}
    objects = []

    ### PROCESSING OBJECT DICTIONARY ###
    for key in object_dict.keys():
        object_list = object_dict[key].split(', ')

        '''
            For each object in the objects list:
                - Use spacy POS tagger to tag POS of each word
                - Remove stop words
                - Find objects using NN (hallway), NNS (stairs)
                  or JJ preceeded NN = JJ_NN (green_door)
        '''
        for item in object_list:
            parsed_text = POS_tagger(item)
            POS_dict = {}
            
            for token in parsed_text:
                if token.is_alpha and str(token) not in stop_words:
                    POS_dict[str(token)] = token.tag_
            # print(POS_dict)
            POS_values = list(POS_dict.values())
            POS_keys = list(POS_dict.keys())
            
            for i in range(len(POS_values)):
                if i == 0 and (POS_values[i] == "NN" or POS_values[i] == "NNS"):
                    objects.append(POS_keys[0] + "_" + key)
                else:
                    # If the current word is NN or NNS, check if preceeded word == JJ 
                    # and succeeded word (if possible) == NN.
                    if POS_values[i] == "NN" or POS_values[i] == "NNS":
                        if i+1 == len(POS_values):
                            if POS_values[i-1] == "JJ":
                                obj = POS_keys[i-1] + "_" + POS_keys[i] + "_" + key
                                objects.append(obj)
                        else:                           
                            if POS_values[i-1] == "JJ":
                                if POS_values[i+1] == "NN":
                                    obj = POS_keys[i-1] + "_" + POS_keys[i] + "_" + POS_keys[i+1] + "_" + key
                                    objects.append(obj)
                                else:
                                    obj = POS_keys[i-1] + "_" + POS_keys[i] + "_" + key
                                    objects.append(obj)

    ### PROCESSING DESCRIPTION DICTIONARY ###
    action_array = []
    for key in description_dict.keys():
        val = description_dict[key].replace('. ', ', ')
        description_list = val.split(', ')

        # get POS for each description
        for desc in description_list:
            parsed_text = POS_tagger(desc)
            POS_dict = {}
            action = ''
            pos = []

            for token in parsed_text:
                if token.is_alpha:
                    POS_dict[str(token)] = token.tag_
                    
            POS_values = list(POS_dict.values())
            POS_keys = list(POS_dict.keys())

            '''
                Capture possible action and requirement by saving VB after MD (can)
                TODO: how to represent 'requirement' in design documentation & retrieval
            '''
            if 'MD' in POS_values:
                start = POS_values.index('MD') + 1
                if 'TO' in POS_values:
                    end = POS_values.index('TO')
                    for i in range(start, end):
                        pos.append(POS_values[i])
                        action = action + POS_keys[i] + ' '

                    if 'NN' in pos or 'NNS' in pos:
                        action += key
                        action_array.append(action)
                    else:
                        # append the smallest position of the NN or NNS
                        temp = []
                        for p_o_s in POS_values:
                            if p_o_s == 'NN' or p_o_s == 'NNS':
                                temp.append(POS_values.index(p_o_s))
                        noun_pos = min(temp)
                        action = action + POS_keys[noun_pos] + ' ' + key
                        action_array.append(action)
                elif 'IN' in POS_values:
                    end = POS_values.index('IN')
                    for i in range(start, end):
                        pos.append(POS_values[i])
                        action = action + POS_keys[i] + ' '

                    if 'NN' in pos or 'NNS' in pos:
                        action += key
                        action_array.append(action)
                    else:
                        # append the smallest position of the NN or NNS
                        temp = []
                        for p_o_s in POS_values:
                            if p_o_s == 'NN' or p_o_s == 'NNS':
                                temp.append(POS_values.index(p_o_s))
                        noun_pos = min(temp)
                        action = action + POS_keys[noun_pos] + ' ' + key
                        action_array.append(action)
                else:
                    action = ' '.join(POS_keys[start:])
                    action += key
                    action_array.append(action)

    final_actions = list(set(action_array))
    #print(final_actions)

    objects = list(set(objects))
    #print(objects)
    rooms = list(object_dict.keys())
    preps = ["middle", "right", "left", "center"]

    '''
        The level knowledge (as described earlier) is represented as a dictionary
        in the following format:
        {
            "hallway": {
                "location": "room-a",
                "rel_position": NULL,
                "pos_action": "go down", "walk back",
                "destination": "room-b",
                "action_req": NULL
            },
            ...
        }
    '''
    knowledge_dict = {}
    for obj in objects:
        obj_knowledge_dict = {}
        for loc in rooms:
            if loc in obj:
                obj = obj.replace('_{0}'.format(loc), '')
                obj_knowledge_dict["location"] = loc
                
        for prep in preps:
            if prep in obj:
                obj_knowledge_dict["rel_position"] = prep
                obj = obj.replace('_{0}'.format(prep), '')

        for action in final_actions:
            obj_ = obj.replace('_', ' ')
            obj_word_list = nltk.word_tokenize(obj_)
            if obj_ in action:
                if obj_knowledge_dict["location"] in action:
                    action = action.replace(obj_knowledge_dict["location"], '')
                    obj_knowledge_dict["pos_action"] = action.strip()
                    break
            else:
                for word in obj_word_list:
                    if word in action:
                        if obj_knowledge_dict["location"] in action:
                            action = action.replace(obj_knowledge_dict["location"], '')
                            obj_knowledge_dict["pos_action"] = action.strip()
                            break
                    else:
                        obj_knowledge_dict["pos_action"] = None
                        continue

        obj_knowledge_dict["destination"] = None
        obj_knowledge_dict["action_req"] = None

        if obj not in knowledge_dict:
            knowledge_dict[obj] = obj_knowledge_dict
        else:
            knowledge_dict[obj]["destination"] = obj_knowledge_dict["location"]
            obj_knowledge_dict["destination"] = knowledge_dict[obj]["location"]
            knowledge_dict[obj+'_2'] = obj_knowledge_dict

    #print(knowledge_dict)
    return knowledge_dict

if __name__ == "__main__":
    obj_dict, desc_dict = description_formatter(desc_text)
    knowledge_representation(obj_dict, desc_dict)
    
#%%
