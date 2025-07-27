import os
import json
import re
from apify_client import ApifyClient

def scrape_linkedin_profile(profile_url: str):
    """
    Scrapes a LinkedIn profile using the public Apify actor 'apimaestro/linkedin-profile-detail'.
    This is the definitive, correct implementation for this specific actor.
    """
    apify_api_key = os.getenv("APIFY_API_KEY")
    if not apify_api_key:
        raise ValueError("APIFY_API_KEY environment variable must be set.")

    # Extract the unique username/id from the LinkedIn URL
    match = re.search(r"linkedin\.com/in/([^/]+)", profile_url)
    if not match:
        raise ValueError("Invalid LinkedIn Profile URL. Could not extract username.")
    username = match.group(1)

    client = ApifyClient(apify_api_key, timeout_secs=120)

    # This actor requires the 'username' field, not 'startUrls'.
    run_input = {
        "username": username,
    }

    print(f"Running DEFINITIVE scraper: apimaestro/linkedin-profile-detail for: {username}")

    # Using .actor() as requested by the user for this specific actor.
    run = client.actor("apimaestro/linkedin-profile-detail").call(run_input=run_input)

    print("Apify actor run finished. Fetching results...")
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

    if not dataset_items:
        return json.dumps({"error": "Scraper returned no data for the given URL."})
        
    return json.dumps(dataset_items[0], indent=2)
