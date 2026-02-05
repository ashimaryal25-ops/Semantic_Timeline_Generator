import spacy
import dateparser
from dateparser.search import search_dates
import streamlit as st
from streamlit_timeline import timeline
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# --- MAIN APP ---
def main():
    st.set_page_config(layout="wide")
    
    # 1. Load Resources (Cached)
    @st.cache_resource
    def load_nlp():
        return spacy.load("en_core_web_trf")
    
    @st.cache_resource
    def load_embedder():
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    nlp = load_nlp()
    model = load_embedder()

    # 2. Input Data
    text = """The company was founded in 1998, starting with just three employees in a small office downtown.
      By June 2000, they had launched their first product, which quickly gained traction among early adopters. 
      Over the next few years, the team expanded, opening a branch in New York in 2003.
      Research and development saw a major boost in 2005, leading to the release of version 2.0 in March 2006. 
      Customer feedback during 2007 highlighted areas for improvement, prompting minor updates throughout that year. 
      The year 2008 brought recognition in the industry, with awards for innovation in software design. 
      Despite challenges in 2009, including market competition and internal restructuring, the company maintained steady growth. 
      A major upgrade was rolled out on August 15, 2010, which included new features and improved security. 
      By 2012, international sales had doubled, especially in Europe and Asia. 
      The team celebrated its 20th anniversary in 2018, reflecting on two decades of innovation. 
      In 2020, remote work became standard practice due to global events, which reshaped internal workflows. 
      Recent improvements were implemented in February 2023, optimizing system performance and user experience. 
      Looking ahead, plans for expansion are scheduled for 2025, aiming to enter emerging markets and develop next-generation technologies."""

    doc = nlp(text)
    event_list = create_event_list(doc)

    if event_list:
        # 3. Clustering Logic
        clustered_list = cluster(event_list, model)

        # 4. Partition Data by Cluster ID
        topic_map = defaultdict(list)
        for e in clustered_list:
            topic_map[e["cluster"]].append(e)

        # 5. UI: TOPIC BUTTONS
        st.header("📂 Discovered Topics")
        st.write("The AI has analyzed the document and found the following themes via centroid selection:")

        cols = st.columns(3)
        for i, (c_id, group) in enumerate(topic_map.items()):
            if c_id == -1:
                topic_label = "Miscellaneous Details"
            else:
                # FIX: Pass 'model' instead of 'summarizer' here
                topic_label = generate_auto_topic(group, model)
            
            with cols[i % 3]:
                if st.button(f"📌 {topic_label}", key=f"topic_btn_{c_id}", use_container_width=True):
                    st.success(f"Cluster Selected: {topic_label}")
                    st.info(f"This group contains {len(group)} related events.")

        st.divider()

        # 6. MASTER TIMELINE
        st.header("🕰️ General Chronological Timeline")
        event_list.sort(key=lambda x: x["date"])
        timeline_data = build_timeline_json(event_list)
        timeline(timeline_data, height=500)
        
    else:
        st.warning("No events found to categorize.")

# --- HELPER FUNCTIONS ---

def generate_auto_topic(grouped_events, nlp):
    """
    Uses POS tagging to turn the centroid sentence into a short title.
    """
    if not grouped_events:
        return "General Topic"
    
    # 1. Use the sentence you already identified as the 'center'
    # (Assuming you still use the Centroid logic to pick the best sentence)
    best_sent = grouped_events[0]["Sentence"] 
    doc = nlp(best_sent)
    
    # 2. Extract the main Noun (Subject) and the main Verb
    keywords = []
    for token in doc:
        # We want the root verb and the main subject nouns
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
            keywords.append(token.text)
            if len(keywords) >= 3: # Keep it strictly to 2-3 words
                break
                
    # 3. Join them into a title
    if keywords:
        return " ".join(keywords).title()
    
    # Fallback to a simple truncation if POS tagging fails
    return " ".join(best_sent.split()[:3]).title() + "..."



def date_time_settings():
    return {
        'DATE_ORDER': 'MDY', 
        'PREFER_DAY_OF_MONTH': 'first',
        'PREFER_MONTH_OF_YEAR': 'first',
        'REQUIRE_PARTS': ['year']
    }

def get_date_score(date_string):
    digit_count = sum(c.isdigit() for c in date_string)
    length = len(date_string)
    return (digit_count, length)

def create_event_list(doc):
    event_list = []
    stgs = date_time_settings()
    for sent in doc.sents:
        date_ents = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
        valid_dates = [d for d in date_ents if any(char.isdigit() for char in d)]
        if not valid_dates: continue

        raw_date_str = max(valid_dates, key=get_date_score)
        found_dates = search_dates(raw_date_str, settings=stgs)
        
        if found_dates:
            best_sub_date = max(found_dates, key=lambda x: get_date_score(x[0]))
            date_obj = best_sub_date[1]
        else:
            date_obj = dateparser.parse(raw_date_str, settings=stgs)

        if not date_obj: continue

        is_ymd = date_obj.day != 1
        is_ym = date_obj.month != 1 and not is_ymd
        
        event_list.append({
            "Sentence": sent.text,
            "type": "YMD" if is_ymd else "YM" if is_ym else "Y",
            "date": date_obj,
        })
    return event_list

def cluster(event_list, model):
    if not event_list: return event_list
    sentences = [e.get("Sentence", "") for e in event_list]
    embeddings = model.encode(sentences)
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
    return event_list



def build_timeline_json(event_list):
    json_events = []
    for event in event_list:
        d = event["date"]
        if event["type"] == "YMD":
            headline = d.strftime("%B %d, %Y")
        elif event["type"] == "YM":
            headline = d.strftime("%B %Y")
        else:
            headline = d.strftime("%Y")

        json_events.append({
            "text": {"headline": headline, "text": event["Sentence"]},
            "start_date": {"year": d.year, "month": d.month, "day": d.day}
        })
    return {"events": json_events}

if __name__ == "__main__":
    main()