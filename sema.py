import spacy
import dateparser
import re
import numpy as np
import hdbscan
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# 1. SETUP
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Error: Run 'python -m spacy download en_core_web_lg' first.")
    exit()

model = SentenceTransformer('all-MiniLM-L6-v2')

text = """The 'Deep Blue' oceanic research initiative officially launched on 2021-01-05, aiming to map the Mariana Trench. 
Construction workers on the Riverside project began installing the steel frame on 06-20-2021. 
By 2021-02-15, the first Deep Blue drone prototype completed its initial pressure-chamber testing. 
Simultaneously, the City Council approved the 'Urban Greenway' project on 2021-01-12 to convert abandoned rails into parks. 
In the financial sector, 'Capital Horizon' released its Q1 market outlook on 2021-02-25. 
The Urban Greenway team began removing old tracks on 06-15-2021, starting with the North End section. 
A massive shipping merger between 'Atlas Marine' and 'Vanguard Freight' was finalized on 2021-04-05. 
However, the Urban Greenway project faced its first hurdle on 03-10-2021 when soil contamination was discovered. 
On the same day, 04-05-2021, the Deep Blue initiative deployed its first surface vessel, the 'RV Explorer.'
By May 20, 2021, Capital Horizon announced a new $500 million 'Sustainability Fund.' 
The City Transit Authority launched the 'Electric Pulse' bus program on 2021-08-12. 
Meanwhile, a cybersecurity audit of the Deep Blue data network on 2021-07-10 revealed no vulnerabilities. 
In September, 09-05-2021, Capital Horizon’s sustainability fund awarded its first grant to a startup in Norway. 
The Deep Blue project hit a major milestone on 10-20-2021, capturing 3D scans of the seafloor. 
Winter brought challenges, as Atlas Marine reported a supply chain bottleneck on 11-15-2021. 
The Urban Greenway project completed its first mile of paved walking paths on 12-05-2021. 
To wrap up 2021, the Electric Pulse bus program reported its 100,000th passenger on 12-28-2021. 
The new year began with Deep Blue’s second phase on 01-10-2022, focusing on biological samples. 
The City Council announced a tax incentive for businesses along the Urban Greenway on 01-25-2022. 
Atlas Marine and Vanguard Freight launched their joint 'Global Port Tracker' app on 03-10-2022. 
Finally, the Electric Pulse program expanded to the suburbs on 04-01-2022, adding charging stations. 
Deep Blue concluded its primary mission on 05-15-2022, returning to port with a decade of data. 
The Urban Greenway officially held its grand opening ribbon-cutting ceremony on 06-01-2022.
"""
doc = nlp(text)
event_list = []

# 2. EXTRACTION & NORMALIZATION (Fixing point 2 & 6)
dt_settings = {'DATE_ORDER': 'YMD', 'PREFER_DAY_OF_MONTH': 'first'}
default_date = datetime(1900, 1, 1) # Fallback for sorting

for sent in doc.sents:
    dt_obj = None
    
    match_numeric = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', sent.text)
    match_text = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})', sent.text, re.IGNORECASE)

    if match_numeric:
        dt_obj = dateparser.parse(match_numeric.group(), settings=dt_settings)
    elif match_text:
        dt_obj = dateparser.parse(match_text.group(), settings=dt_settings)
    
    # If no specific date found, we skip to maintain a clean timeline
    if dt_obj:
        event_list.append({
            "sentence": sent.text.strip(),
            "date": dt_obj, # Strictly datetime object now
            "entities": [(ent.text, ent.label_) for ent in sent.ents],
            "cluster": -1 # Default to Misc
        })

# 3. CLUSTERING (Fixing point 4: Metric & Normalization)
if event_list:
    embeddings = model.encode([e["sentence"] for e in event_list])
    # Normalizing + Euclidean is mathematically equivalent to Cosine Distance
    emb_norm = normalize(embeddings).astype('float64')

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2, 
        min_samples=1, 
        cluster_selection_epsilon=0.5, 
        metric='euclidean' 
    )
    labels = clusterer.fit_predict(emb_norm)
    
    for i, label in enumerate(labels):
        event_list[i]["cluster"] = int(label)

# 4. GROUPED OUTPUT (Fixing point 5: Cluster-to-story interpretation)
# Sort entire list by date first
event_list.sort(key=lambda x: x["date"])

print("\n--- STORYLINE RECONSTRUCTION ---")
unique_clusters = sorted(list(set([e["cluster"] for e in event_list])))

for cluster in unique_clusters:
    topic_name = f"TOPIC {cluster}" if cluster != -1 else "MISCELLANEOUS"
    print(f"\n>> {topic_name}")
    for event in event_list:
        if event["cluster"] == cluster:
            print(f"   {event['date'].strftime('%Y-%m-%d')} | {event['sentence']}")