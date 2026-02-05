import re
from transformers import pipeline

# Load once
summarizer = pipeline("summarization", model="t5-small")

def extract_events(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    events = []

    for s in sentences:
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', s)
        if years:
            events.append({
                "year": years[0],
                "sentence": s
            })

    return events


def generate_title(sentences):
    joined = " ".join(sentences)

    prompt = f"Generate a short topic title:\n{joined}"

    result = summarizer(
        prompt,
        max_length=10,
        min_length=3,
        do_sample=False
    )[0]["summary_text"]

    return result.title()


# ---------- Example ----------
text = """
The company was founded in 1998 with three employees.
In 2003, a New York office was opened.
A major upgrade was released in 2010.
Remote work became standard in 2020.
"""

events = extract_events(text)

title = generate_title([e["sentence"] for e in events])

print("Topic:", title)
for e in events:
    print(e["year"], "->", e["sentence"])
