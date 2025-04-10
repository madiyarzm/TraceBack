from dotenv import load_dotenv
from serpapi import GoogleSearch
import os

def get_related_articles(article, num = 5):
    #loading environmental variables in safe way
    load_dotenv()
    api_key = os.getenv("SERPAPI_KEY")

    #creating search parameters in a dictionary
    test_parameters = {
        "engine": "google",
        "q": article,
        #"q": "A Fargo, North Dakota, man was arrested for clearing snow with a flamethrower.",
        "api_key": api_key,
        "num": num
    }

    #passing it to searching function 
    search_request = GoogleSearch(test_parameters)

    #parses JSON response, and returns in dictionary format
    results = search_request.get_dict()

    #getting organic results without adds, not needed data
    needed_results = results.get("organic_results", [])

    articles = []
    
    #looping through to get needed data:
    for res in needed_results:
        """
        print("Title", res.get("title"))
        print("URL:", res.get("link"))
        print("Snippet", res.get("snippet"))
        print("Date:", res.get("date", "Not available"))

        
        print("-" * 40)
        """
        
        articles.append({
            "title": res.get("title"),
            "url": res.get("link"),
            "snippet": res.get("snippet"),
            "date": res.get("date", "Not available")  
        })
    
    return articles

