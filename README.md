# Semantic Timeline Generator ??

**Live Demo:** [View the App on Streamlit](https://semantictimelinegenerator-a7fwnujyubkvpcimsrycb6.streamlit.app/)

The **Semantic Timeline Generator** is an advanced NLP tool that extracts events and dates from unstructured text and organizes them into an interactive, chronological timeline. Version 3 introduces **Semantic Topic Discovery**, allowing the tool to categorize events by theme rather than just time.

## ? New in Version 3: Semantic Logic

* **Advanced Semantic Clustering:** Implements **HDBSCAN** and **SentenceTransformers** (all-MiniLM-L6-v2) to mathematically group related events into topics based on their semantic meaning.
* **Rule-Based Topic Labeling:** Automatically generates descriptive titles for each cluster by extracting key noun phrases directly from the source text, ensuring labels are grounded and date-free.
* **Interactive Filtering:** A new "Discovered Topics" dashboard allows users to toggle specific themes on or off, filtering the timeline in real-time.
* **Transformer-Powered Extraction:** Utilizes spaCy's high-accuracy en_core_web_trf (RoBERTa) model for state-of-the-art entity recognition.

## ??? Core Features

* **Chronological Intelligence:** Automatically detects date granularity (Year, Month, or Day) and sorts events accordingly.
* **Robust Date Parsing:** Handles complex date ranges and varied formats using dateparser.
* **Interactive Visualization:** Renders history through a dynamic, scrollable timeline component powered by streamlit-timeline.

## ?? Installation & Usage

1. **Clone the Repo:**
   \git clone https://github.com/ashimaryal25-ops/Semantic_Timeline_Generator.git\
2. **Install Dependencies:**
   \pip install -r requirements.txt\
3. **Run Locally:**
   \streamlit run semantic_timeline_3.py\

## ?? Technical Stack

* **NLP:** spacy (en_core_web_trf), dateparser
* **ML/Math:** hdbscan, scikit-learn, sentence-transformers
* **Frontend:** streamlit, streamlit-timeline
* **Engine:** torch (PyTorch)
