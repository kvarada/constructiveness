__author__ = 'simranjitsingh'

'''
This code is part of the implementation of the following paper: 
  
D. Park, S. Sachar, N. Diakopoulos, and N. Elmqvist. 
Supporting Comment Moderators in Identifying High Quality Online News Comments. 
Proc. Conference on Human Factors in Computing Systems (CHI). May, 2016. [PDF]

We use some of the features used in this work for constructiveness.    
'''

from Normalize_Contraction import NormalizeContraction
from nltk.tokenize import SpaceTokenizer, WhitespaceTokenizer
from bs4 import BeautifulSoup
import string
import re

def CleanAndTokenize(text):
    # Strip URLs and replace with token "URLURLURL"
    r = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    text = re.sub(r, " URLURLURL", text)
    # Strip html tags
    soup = BeautifulSoup(text, "lxml")
    for tag in soup.findAll(True):
        tag.replaceWithChildren()
        text = soup.get_text()
    # Normalize everything to lower case
    text = text.lower()
    # Strip line breaks and endings \r \n
    r = re.compile(r"[\r\n]+")
    text = re.sub(r, "", text)
    # get rid of em dashes
    # table = {
    #     ord(u'\u2018') : u"'",
    #     ord(u'\u2019') : u"'",
    #     ord(u'\u201C') : u'"',
    #     ord(u'\u201d') : u'"',
    #     ord(u'\u2026') : u'',
    #     ord(u'\u2014') : u'',
    # }
    # text = text.translate(table)

    # Normalize contractions
    # e.g. can't => can not, it's => it is, he'll => he will
    text = NormalizeContraction(text)

    # Strip punctuation (except for a few)
    punctuations = string.punctuation
    # includes following characters: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    excluded_punctuations = ["$", "%"]
    for p in punctuations:
        if p not in excluded_punctuations:
            text = text.replace(p, " ")

    # Condense double spaces
    text = text.replace("  ", " ")

    # Tokenize the text
    tokenizer = WhitespaceTokenizer()
    text_tokens = tokenizer.tokenize(text)
    return text_tokens
