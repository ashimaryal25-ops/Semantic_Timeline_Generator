import spacy
import dateparser
from dateparser.search import search_dates
import streamlit as st
from streamlit_timeline import timeline

def main():
    st.set_page_config(layout="wide")
    
    @st.cache_resource
    def load_nlp():
        return spacy.load("en_core_web_trf")
    
    nlp = load_nlp()

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
    
    #gets events and creates timline
    if event_list:
        event_list.sort(key=lambda x: x["date"])
        create_timeline(event_list)
    else:
        st.warning("No events found")


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
    timeline_data = {"events": []}
    #formats dateparser dates to match the granulity
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
        
        # Clean None values
        #  Create a temporary dictionary to hold the clean data
        cleaned_date_info = {}

        for key, value in data_item["start_date"].items():
    

            if value is not None:
                cleaned_date_info[key] = value

        # Overwrite the original dictionary with the cleaned version
        data_item["start_date"] = cleaned_date_info

        # Add the processed item to the timeline list
        timeline_data["events"].append(data_item)

    timeline(timeline_data, height=500)

if __name__ == "__main__":
    main()
    