# Semantic Timeline Generator 🚀

**Live Demo:** [View the App on Streamlit](https://semantictimelinegenerator-a7fwnujyubkvpcimsrycb6.streamlit.app/)

The **Semantic Timeline Generator** is an advanced NLP tool that extracts events and dates from unstructured text and organizes them into an interactive, chronological timeline. Version 3 introduces **Semantic Topic Discovery**, allowing the tool to categorize events by theme rather than just time.

## 🚀 Key Features

* **Event Extraction:** Automatically finds dates and events in raw text using Transformer-based NLP (spaCy \`en_core_web_trf\`).
* **PDF Support:** Upload a selectable-text PDF and the app extracts its text before generating topics and timeline events.
* **Topic Grouping:** Uses **HDBSCAN clustering** to mathematically group related events into categories so you don't just have one long list.
* **Smart Sorting:** Handles different date types (Years vs. Months vs. Days) and keeps everything in the right order.
* **Interactive Timeline:** Generates a scrollable, clickable timeline where you can see the story of the text visually.
* **Topic Filtering:** A dashboard that lets you toggle specific subjects on or off to clean up the view.
* **Auto-Labeling:** Automatically gives names to each group of events based on the text inside them.

## ⚙️ Installation & Usage

1. **Clone the Repo:**
   git clone https://github.com/ashimaryal25-ops/Semantic_Timeline_Generator.git

2. **Install Dependencies:**
   pip install -r requirements.txt

3. **Run Locally:**
   streamlit run semantic_timeline_3.py

## 📦 Technical Stack

* **NLP:** spacy (en_core_web_trf), dateparser
* **File Parsing:** pypdf
* **ML/Math:** hdbscan, scikit-learn, sentence-transformers
* **Frontend:** streamlit, streamlit-timeline
* **Engine:** torch (PyTorch)
