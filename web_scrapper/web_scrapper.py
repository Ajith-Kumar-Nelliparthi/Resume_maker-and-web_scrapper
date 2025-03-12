# Import libraries
import numpy as np 
import os
import json
import re 
from bs4 import BeautifulSoup
import requests
import spacy

class CleanTextWebpage:
    def __init__(self, text):
        self.text = text

    def clean_text(self):
        text = re.sub(r'\n+', '\n', self.text)      # Replace multiple new lines with a single new line
        text = re.sub(r'\s+', ' ', text)            # Replace multiple spaces with a single space
        text = re.sub(r'[^\x00-\x7F]+', '', text)   # Remove non-ASCII characters
        return text.strip()

# Function to scrape the website using URL
class ScrapeWebsite:
    def __init__(self, url):
        self.url = url
        
    def scrap(self):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for tag in soup(['style', 'script', 'footer', 'aside', 'nav']):
                tag.decompose()

            # Extract meaningful content
            main_content = soup.find("article") or soup.find("main") or soup.body
            text = main_content.get_text(separator=" ", strip=True) if main_content else "No meaningful content found."
            
            # Clean the text
            cleaned_text = CleanTextWebpage(text).clean_text()

            data = {
                'title': soup.title.string if soup.title else 'No Title',
                'text': cleaned_text
            }

            json_filename = "scraped_data.json"
            with open(json_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            return cleaned_text  
        except requests.exceptions.RequestException as e:
            return f"Request error: {e}"
    
def search_person(person_name):
    API_KEY = "your_api_key_here"  
    search_url = "https://serpapi.com/search"
    
    params = {
        "q": person_name,
        "hl": "en",
        "gl": "us",
        "api_key": API_KEY
    }

    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        filtered_results = [
            result for result in results if not any(
                site in result['link'] for site in ["instagram.com", "twitter.com", "facebook.com", "fandom.com"]
            )
        ]

        if filtered_results:
            print(f"\nTop results for {person_name}:")
            for i, result in enumerate(filtered_results[:3], 1):           # Show top 3 filtered results
                print(f"{i}. {result['title']}\n   {result['link']}\n")
            return filtered_results[0]['link']                             # Return the best non-social media link
        else:
            print("\nNo suitable results found (filtered out social media and fan pages).")
            return None
    else:
        print("\nFailed to fetch search results.")
        return None




if __name__ == '__main__':
    url = input("Enter the URL of the website you want to scrape: ")
    scraper = ScrapeWebsite(url)    # Create an instance of the ScrapeWebsite class
    cleaned_text = scraper.scrap()  # Call the scrap method and get cleaned text
    print(f"\nThe website content scrapped and cleaned.\n")

    # Now perform entity recognition
    from entity import EntityRecognition
    entity_recognition = EntityRecognition(cleaned_text)  # Pass the cleaned text
    entities = entity_recognition.extract_entities()

    print("\nPersons found:", entities["persons"])
    print("\nOrganizations found:", entities["organizations"])

    
    # Find persons in the URL if user wants
    if input("\nDo you want to look up persons found in the URL? (yes/no) ").lower() == "yes":
        person_name = input("\nEnter the person's name you want to search: ")
        
        if person_name in entities["persons"]:
            print(f"\nSearching information about {person_name}...\n")
            search_url = search_person(person_name)

            if search_url:
                print(f"\nScraping details from {search_url}...\n")
                person_scraper = ScrapeWebsite(search_url)
                cleaned_person_text = person_scraper.scrap()

                filename = f"{person_name.replace(' ', '_')}.json"
                print(f"\nDetails saved in '{filename}'.")
                with open(filename, 'w') as json_file:
                    json.dump({"name": person_name, "text": cleaned_person_text}, json_file, indent=4)
        else:
            print("\nThe entered name was not found in the extracted persons list.")
    else:
        print("\nSkipping person lookup. Process complete.")



'''
Task: Enhanced Web_Scrapping
Description:
1. Designed a python code for web_scrapping using Beautifulsoup, spacy, json and requests libraries.
2. Used SerpApi for searching persons in the web.
3. Extracted entities from the scrapped text using spacy library.
4. Saved the extracted entities in a json file.
5. Added a feature to search for persons in the scrapped text and save their details in a json
6. Used a class based approach for the code.
7. search_engine function is used to search for persons in the web using SerpApi. It will return the top 3 website pages

SerpApi is a tool designed to scrape and parse search results from various search engines like Google, Bing, Yahoo, and more. 
It provides structured data in JSON format, making it easier for developers to work with search engine results. 
Here are some key features:
    Real-Time Results: Each API request runs immediately, mimicking human behavior to ensure accurate results.
    Global Reach: You can get search results from any location worldwide using the "location" parameter.
    Advanced Features: It includes CAPTCHA-solving technology, browser clusters, and structured SERP data.
    Integration: Supports multiple programming languages like Python, JavaScript, Ruby, and more.
'''

            