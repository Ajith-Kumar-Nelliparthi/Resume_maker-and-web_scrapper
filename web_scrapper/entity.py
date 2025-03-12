import spacy
import re
from collections import Counter

class EntityRecognition:
    def __init__(self, text):
        self.text = text 
        self.nlp = spacy.load("en_core_web_sm") 
    
    # Function for Named Entity Recognition (NER)
    def extract_entities(self):
        if not isinstance(self.text, str):  
            raise ValueError("Input text must be a string")

        doc = self.nlp(self.text)
        persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']

        # Filter organizations using regex
        org_pattern = re.compile(r'\b(Inc\.|LLC|Corp\.|Limited|Company|Co\.|Group|Foundation|University|Institute)\b', re.IGNORECASE)
        filtered_orgs = [org for org in set(organizations) if org_pattern.search(org)]

        return {
            "persons": list(set(persons)),
            "organizations": list(set(filtered_orgs))
        }
