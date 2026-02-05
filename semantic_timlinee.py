import spacy
import dateparser
from dateparser.search import search_dates
import streamlit as st
from streamlit_timeline import timeline
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import defaultdict
def main():
    st.set_page_config(layout="wide")
    
    @st.cache_resource
    def load_nlp():
        return spacy.load("en_core_web_trf")
    
    @st.cache_resource
    def load_summarizer():
    # 't5-small' is very fast and great for short titles
        return pipeline("summarization", model="t5-small")
    
    nlp = load_nlp()
    summarizer = load_summarizer()

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

    event_list = get_event_list(text, nlp)

    @st.cache_resource
    def load_embed_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
    #clustered timeline by topics:
    model = load_embed_model()

    if "active_clusters" not in st.session_state:
        st.session_state["active_clusters"] = set()


    if event_list:
        # 1. Enrichment: Clustering is already done by your cluster() function
        clustered_list = get_clustered_list(event_list, model)

        # 2. Partition the data by Cluster ID
        topic_map = defaultdict(list)
        for e in clustered_list:
            topic_map[e["cluster"]].append(e)

        # 3. THE UI: Topic Buttons

            # 3. THE UI: Topic Buttons
        st.header("Discovered Topics")

        for c_id, group in topic_map.items():

            if c_id == -1:
                topic_label = "Miscellaneous Details"
            else:
                topic_label = generate_auto_topic(group, summarizer)

    # Multi-select toggle
            checked = c_id in st.session_state.get("active_clusters", set())
            new_checked = st.checkbox(f"📌 {topic_label}", value=checked, key=f"topic_chk_{c_id}")

    # Update session_state
            if new_checked:
                st.session_state["active_clusters"].add(c_id)
            else:
                st.session_state["active_clusters"].discard(c_id)

    # Show events
            if c_id in st.session_state["active_clusters"]:
                for e in group:
                    st.write("•", e["Sentence"])

    else:
        st.warning("No events found to categorize.")
    
    #general timeline showing all events, gets events and creates timline
    if event_list:
        event_list.sort(key=lambda x: x["date"])
        data_items = create_timeline(event_list)
        display_timeline(data_items)
        
    else:
        st.warning("No events found")

@st.cache_data
def get_event_list(text, _nlp):
    doc = _nlp(text)
    return create_event_list(doc)

# Cache clustering
@st.cache_data
def get_clustered_list(event_list, _model):
    return cluster(event_list, _model)

def date_time_settings():
    return {
        'DATE_ORDER': 'MDY', 
        'PREFER_DAY_OF_MONTH': 'first',
        'PREFER_MONTH_OF_YEAR': 'first',
        'REQUIRE_PARTS': ['year']
    }


#gets date score based on digit count and length of the date, if digit is a tie it will compare the length
def get_date_score(date_string):
    digit_count = sum(c.isdigit() for c in date_string)
    length = len(date_string)
    return (digit_count, length)

def create_event_list(doc):
    event_list = []
    stgs = date_time_settings()

    for sent in doc.sents:
        date_ents = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
        
        #  Remove pure text dates
        valid_dates = []

        for d in date_ents:
        # Check if the current date string contains at least one digit
            if any(char.isdigit() for char in d):
                valid_dates.append(d)
        
        if not valid_dates:
            continue

        # Pick the richest date string
        raw_date_str = max(valid_dates, key=get_date_score)

        # Handle ranges - search inside the string like between March 2000 and August 2005
        found_dates = search_dates(raw_date_str, settings=stgs)
        
        if found_dates:
            best_sub_date = max(found_dates, key=lambda x: get_date_score(x[0]))
            date_str = best_sub_date[0]
            date_obj = best_sub_date[1]
        else:
            date_str = raw_date_str
            date_obj = dateparser.parse(date_str, settings=stgs)

        if not date_obj:
            continue

        # Granularity detection
        year_month_day = year_month = year = False
        if date_obj.day != 1:
            year_month_day = True
        elif date_obj.month != 1:
            year_month = True
        else:
            year = True
        event = {
            "Sentence": sent.text,
            "type": "YMD" if year_month_day else "YM" if year_month else "Y",
            "date": date_obj,
        }
        event_list.append(event)
    
    return event_list    

def create_timeline(event_list):
    #formats dateparser dates to match the granulity
    all_items = []  # Initialize a list to hold all events
    
    for event in event_list:
        if event["type"] == "YMD":
            headline_date = event["date"].strftime("%B %d, %Y")
        elif event["type"] == "YM":
            headline_date = event["date"].strftime("%B %Y")
        else:
            headline_date = event["date"].strftime("%Y")

        data_item = {
            "text": {
                "headline": headline_date,
                "text": event["Sentence"]
            },
            "display_date": "Date:",
            "start_date": {
                "year": event["date"].year,
                "month": event["date"].month,
                "day": event["date"].day
            }
        }
        all_items.append(data_item) # Add this item to the list
        
    return all_items # Return the full list, not just one dict

def display_timeline(data_items):   
    timeline_data = {"events": []}    
    
    # data_items is now a LIST, so we must loop through it
    for item in data_items: 
        cleaned_date_info = {}

        # Access "start_date" from the individual 'item', not the list 'data_items'
        for key, value in item["start_date"].items():
            if value is not None:
                cleaned_date_info[key] = value

        # Overwrite the date info for this specific item
        item["start_date"] = cleaned_date_info

        # Add this specific item to the final events list
        timeline_data["events"].append(item)

    # Finally, pass the dictionary containing the list of events to the timeline component
    timeline(timeline_data, height=500)


    #clustering

def cluster(event_list, model):
    # Check if we have enough data to cluster
    if not event_list:
        return event_list

    # 1. Vectorize sentences (lowercase 'sentence' key must match your create_event_list)
    # Note: Use e.get("Sentence") if you used uppercase in the previous function
    sentences = [e.get("Sentence", "") for e in event_list]
    embeddings = model.encode(sentences)

    # 2. Normalize for Cosine Similarity proxy
    emb_norm = normalize(embeddings).astype('float64')

    # 3. Correct way to pass parameters to the Class
    clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2, 
    min_samples=1, 
    cluster_selection_epsilon=0.5, 
    metric='euclidean'
    )
    
    # 4. Generate the labels
    labels = clusterer.fit_predict(emb_norm)
    
    # 5. Map back to the list (Inside the scope where labels exists)
    for i, label in enumerate(labels):
        event_list[i]["cluster"] = int(label)

    return event_list

def generate_auto_topic(grouped_events, summarizer):
    combined_text = " ".join(e["Sentence"] for e in grouped_events)

    result = summarizer(combined_text, max_length=10, min_length=3, do_sample=False)
    topic = result[0]['summary_text']

    return topic.title()

if __name__ == "__main__":
    main()
    
