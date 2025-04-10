from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("SERPAPI_KEY")

test_parameters = {
    "engine": "google",
    "q": "A Fargo, North Dakota, man was arrested for clearing snow with a flamethrower.",
    "api_key": api_key,
    "num": 5
}


