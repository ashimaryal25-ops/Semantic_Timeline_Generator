import spacy
import dateparser

nlp = spacy.load("en_core_web_sm")  # Load the specified spaCy model
text = """Company A launched a product on January 5, 2023. Company B filed a patent on Jan 3, 2023."""

doc = nlp(text)

event_list = []

for sent in doc.sents:
    date = None

    for ent in sent.ents:
        if ent.label_== "DATE":
            date = dateparser.parse(ent.text)
            break

    event = {
        "sentence": sent.text,
        "date": date,
        "entities": [(ent.text, ent.label_) for ent in sent.ents]

    }    
    event_list.append(event)

for event in event_list:
    print(event)