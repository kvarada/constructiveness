import spacy
import numpy as np

nlp = spacy.load('en')

class CommentLevelFeatures():
    '''
    Get sophisticated comment-level features
    '''
    def __init__(self, text):
        '''
        :param text:
        '''
        self.text = text
        self.doc = nlp(text)
        self.entities = {}

    def get_pos_seq(self):
        '''
        '''
        return " ".join([token.pos_ for token in self.doc])
        #return [pos for pos in self.doc]
        
    def get_named_entities(self):
        '''
        :return:
        '''
        for ent in self.doc.ents:
            if ent.label_ in self.entities.keys():
                self.entities[ent.label_].append(ent)
            else:
                self.entities[ent.label_] = [ent]
        return self.entities

    def get_named_entity_counts(self, ne_type='all'):
        '''
        :param ne_type:
        :return:
        '''
        entity_count = 0
        if ne_type in self.entities.keys():
            entity_count += len(self.entities[ne_type])
            return entity_count

        # Count all entities
        for (entity_type, entities) in self.entities.items():
            if ne_type.startswith('all'):
                entity_count += len(entities)
        return entity_count
    
    def get_sentences(self):
        '''
        '''
        return [sent for sent in self.doc.sents]
    
    def get_sentence_counts(self):
        '''
        :return:
        '''
        sents = [sent for sent in self.doc.sents]
        return len(sents)

    def average_nwords_per_sentence(self):
        '''
        :return:
        '''
        sents = [sent for sent in self.doc.sents]
        return round(np.mean([len(sent) for sent in sents]),3)

if __name__=="__main__":
    cf = CommentLevelFeatures(u'London is a big city in the United Kingdom. It is one of my favourite cities. Mr. Douglas Beaton is from Scotland.')
    print('POS tagged: ', cf.get_pos_seq())
    entities = cf.get_named_entities()
    print(entities)
    print(cf.get_named_entity_counts('PERSON'))
    print('Number of sentences: ', cf.get_sentence_counts())
    print('Average number of words per sentence: ', cf.average_nwords_per_sentence())