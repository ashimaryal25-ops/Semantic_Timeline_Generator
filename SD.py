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
    
    # ----------------- Load Models -----------------
    @st.cache_resource
    def load_nlp():
        return spacy.load("en_core_web_trf")
    
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="t5-small")
    
    nlp = load_nlp()
    summarizer = load_summarizer()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # ----------------- Input Text -----------------
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
        # ----------------- Cluster Events -----------------
        clustered_list = cluster(event_list, model)

        # ----------------- Group by Cluster -----------------
        topic_map = defaultdict(list)
        for e in clustered_list:
            topic_map[e["cluster"]].append(e)

        # ----------------- UI: Topic Buttons -----------------
        st.header("Discovered Topics")
        st.write("Following themes have been found:")

        cols = st.columns(3)

        for i, (c_id, group) in enumerate(topic_map.items()):
            # Handle outliers (-1) vs real clusters
            if c_id == -1:
                topic_label = "Miscellaneous Details"
            else:
                topic_label = generate_auto_topic(group)

            with cols[i % 3]:
                # Button to select cluster
                if st.button(f"📌 {topic_label}", key=f"topic_btn_{c_id}"):
                    st.session_state["active_cluster"] = c_id

                # Display events for the active cluster directly under the button
                if st.session_state.get("active_cluster") == c_id:
                    st.markdown(f"**Events for {topic_label}:**")
                    for e in group:
                        st.write("•", e["Sentence"])

    else:
        st.warning("No events found to categorize.")

    # ----------------- Timeline -----------------
    if event_list:
        event_list.sort(key=lambda x: x["date"])
        data_items = create_timeline(event_list)
        display_timeline(data_items)
    else:
        st.warning("No events found")

# ----------------- Helper Functions -----------------

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
        if not valid_dates:
            continue

        raw_date_str = max(valid_dates, key=get_date_score)
        found_dates = search_dates(raw_date_str, settings=stgs)
        if found_dates:
            best_sub_date = max(found_dates, key=lambda x: get_date_score(x[0]))
            date_obj = best_sub_date[1]
        else:
            date_obj = dateparser.parse(raw_date_str, settings=stgs)

        if not date_obj:
            continue

        # Determine granularity
        year_month_day = year_month = year = False
        if date_obj.day != 1:
            year_month_day = True
        elif date_obj.month != 1:
            year_month = True
        else:
            year = True

        event_list.append({
            "Sentence": sent.text,
            "type": "YMD" if year_month_day else "YM" if year_month else "Y",
            "date": date_obj
        })

    return event_list

def create_timeline(event_list):
    data_items = {"events": []}
    for event in event_list:
        if event["type"] == "YMD":
            headline_date = event["date"].strftime("%B %d, %Y")
        elif event["type"] == "YM":
            headline_date = event["date"].strftime("%B %Y")
        else:
            headline_date = event["date"].strftime("%Y")

        item = {
            "text": {"headline": headline_date, "text": event["Sentence"]},
            "display_date": "Date:",
            "start_date": {
                "year": event["date"].year,
                "month": event["date"].month,
                "day": event["date"].day
            }
        }
        # Clean None values
        item["start_date"] = {k: v for k, v in item["start_date"].items() if v is not None}
        data_items["events"].append(item)
    return data_items

def display_timeline(data_items):
    timeline(data_items, height=500)

def cluster(event_list, model):
    if not event_list:
        return event_list

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

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_trf")

nlp = load_nlp()  # load once globally

def generate_auto_topic(grouped_events):
    from collections import Counter

    # Combine all sentences in the cluster
    combined_text = " ".join([e["Sentence"] for e in grouped_events])
    doc = nlp(combined_text)
    
    # Extract nouns and proper nouns as keywords
    keywords = [token.text.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]

    # Count frequency
    most_common = [word for word, freq in Counter(keywords).most_common(4)]

    # Title-case and join
    title = " ".join(most_common).title()
    
    # Ensure at least 2 words
    if len(most_common) < 2:
        title = " ".join(combined_text.split()[:2]).title()
    
    return title

# ----------------- Run -----------------
if __name__ == "__main__":
    main()
