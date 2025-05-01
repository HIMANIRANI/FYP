import requests
from bs4 import BeautifulSoup
import json
import time

# Base URL of the stock brokers page
BASE_URL = "https://sebon.gov.np/intermediaries/stock-brokers?page={}"

# Function to scrape a single page
def scrape_page(page_num):
    url = BASE_URL.format(page_num)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page {page_num}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    brokers = []
    
    # Find all broker cards
    broker_cards = soup.find_all("div", class_="card")
    
    for card in broker_cards:
        broker_info = {}
        
        # Extract broker name
        name_tag = card.find("h1")
        broker_info["name"] = name_tag.text.strip() if name_tag else "N/A"
        
        # Extract Address and Phone details
        info_wrappers = card.find_all("div", class_="info-wrapper")
        for info in info_wrappers:
            text = info.get_text(strip=True, separator=" ").lower()
            if "address" in text:
                broker_info["address"] = text.split("address:")[-1].strip()
            elif "phone" in text:
                broker_info["phone"] = text.split("phone:")[-1].strip()
        
        # Extract Contact Person Details
        contact_div = card.find("div", class_="col-md-6 col-xs-12 col-sm-12")
        if contact_div:
            contact_details = contact_div.get_text(strip=True, separator=" ").lower()
            if "name:" in contact_details:
                broker_info["contact_person"] = contact_details.split("name:")[-1].split("email:")[0].strip()
        
        brokers.append(broker_info)
    
    return brokers

# Function to scrape all pages
def scrape_all_brokers():
    all_brokers = []
    page_num = 1

    while True:
        print(f"Scraping page {page_num}...")
        brokers = scrape_page(page_num)
        
        if not brokers:  # Stop if no brokers found (last page reached)
            break
        
        all_brokers.extend(brokers)
        page_num += 1
        time.sleep(1)  # Add delay to avoid getting blocked
    
    return all_brokers

# Run scraper and save data to JSON
brokers_data = scrape_all_brokers()

# Save to JSON file
with open("brokers_data.json", "w", encoding="utf-8") as f:
    json.dump(brokers_data, f, indent=4, ensure_ascii=False)

print("Scraping completed! Data saved to brokers_data.json")
