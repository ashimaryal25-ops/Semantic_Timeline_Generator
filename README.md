# Semantic Timeline Generator 🚀

**Live Demo:** [View the App on Streamlit](https://semantictimelinegenerator-a7fwnujyubkvpcimsrycb6.streamlit.app/)

The **Semantic Timeline Generator** is an advanced NLP tool that extracts events and dates from unstructured text and organizes them into an interactive, chronological timeline. Version 3 introduces **Semantic Topic Discovery**, allowing the tool to categorize events by theme rather than just time.

## 🚀 Key Features

* **Semantic Topic Discovery:** Uses HDBSCAN and SentenceTransformers to mathematically cluster related events into logical themes.
* **Context-Aware Extraction:** Leverages spaCy’s Transformer models (en_core_web_trf) for high-accuracy entity and event recognition.
* **Intelligent Date Granularity:** Automatically detects and sorts events by Year, Month, or Day.
* **Interactive Filtering:** Allows users to toggle specific topics to filter the timeline in real-time.
* **Streamlit Visualization:** A clean, web-based interface for interactive data exploration.

## ⚙️ Installation & Usage

1. **Clone the Repo:**
   \git clone https://github.com/ashimaryal25-ops/Semantic_Timeline_Generator.git

2. **Install Dependencies:**
   \pip install -r requirements.txt\

3. **Run Locally:**
   \streamlit run semantic_timeline_3.py\

## 📦 Technical Stack

* **NLP:** spacy (en_core_web_trf), dateparser
* **ML/Math:** hdbscan, scikit-learn, sentence-transformers
* **Frontend:** streamlit, streamlit-timeline
* **Engine:** torch (PyTorch)
