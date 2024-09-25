import arxiv
from openai import AsyncOpenAI
import pandas as pd
from datetime import datetime, timedelta
import pytz
import asyncio
import csv
import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import aiohttp
import re
from urllib.parse import urlparse, urljoin
import aiohttp

# Use environment variable for OpenAI API key
openAI_key = os.environ.get('OPENAI_API_KEY')

# Check if the API key is set
if not openAI_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = AsyncOpenAI(api_key=openAI_key)

def get_arxiv_papers():
    # Get papers from the last 24 hours
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=7)
    start_date = start_date.replace(tzinfo=pytz.UTC)  # Ensure timezone awareness
    end_date = end_date.replace(tzinfo=pytz.UTC)      # Ensure timezone awareness
    
    search = arxiv.Search(
        query="cat:cs.*",  # This will include all computer science categories
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    try:
        for result in search.results():
            # Ensure the published date is timezone-aware
            published_date = result.published.replace(tzinfo=pytz.UTC)
            print(published_date)
            
            papers.append({
                'title': result.title,
                'abstract': result.summary,
                'url': result.entry_id,
                'published': published_date,
                'content': result.pdf_url,  # URL to the full PDF content
                'authors': ', '.join(author.name for author in result.authors)  # List of authors
            })
    except arxiv.UnexpectedEmptyPageError:
        print("Encountered an empty page. Continuing with papers collected so far.")
    
    print(f"Collected {len(papers)} papers")
    return papers

async def classify_paper(abstract):
    domains = ["Mechanistic Interpretability", "Model Routing", "Computer Vision", 
               "Reinforcement Learning", "Finetuning LLMs", "LLM Compression", "Speech Recognition", "Prompting Strategies", "Robotics", 
               "AI Ethics", "Alignment and Safety", "Multi-Modality", "Inference Optimization ", "Large Language Models", "Generative AI", "Not AI", "Other AI"]
    
    prompt = f"""Classify the following research paper abstract into one of these domains: {', '.join(domains)}
    
    If it is not about AI, return "Not AI"

    Abstract: {abstract}
    Provide your answer in the following format:
    DOMAIN: [Selected Domain]
    """
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    message = response.choices[0].message.content

    return message

async def summarize_paper(abstract):
    prompt = f"""Summarize the following research paper abstract in 2-3 sentences:
    {abstract}
    """
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    
    return response.choices[0].message.content.strip()

async def classify_impact(paper):
    prompt = f"""Evaluate the impact and significance of this research paper. Consider factors such as:
    1. The reputation and affiliation of the authors
    2. The novelty and potential influence of the research
    3. The relevance to current AI challenges or advancements
    4. The potential for practical applications or theoretical breakthroughs

    Be critical and set a high bar. Many papers are incremental or have limited impact.

    Title: {paper['title']}
    Authors: {paper['authors']}
    Abstract: {paper['abstract']}

    Provide your answer in the following format:
    IMPACT: [High/Medium/Low]
    REASON: [Brief explanation]
    """
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    
    return response.choices[0].message.content.strip()
async def get_author_email(author_name, session, api_key="AIzaSyDvCoej_7nf7aGr0dxKoiPim32a0vnYMps", search_engine_id="017576662512468239146:omuauf_lfve"):
    try:
        # Google Custom Search API URL
        search_url = "https://www.googleapis.com/customsearch/v1"
        query = f"{author_name} blog"
    
        # Parameters for the search API request
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'num': 3  # Limit the number of search results to 3 for now
        }

        # Make the request to the Google Search API
        async with session.get(search_url, params=params) as response:
            search_results = await response.json()

            # Check if the search results contain URLs
            if 'items' not in search_results:
                print("No items found")
                return None

            # Extract the URLs from search results
            urls = [item['link'] for item in search_results['items']]

            print(urls)

            # Loop through the URLs to find emails
            for url in urls:
                

                # Skip PDF files
                if url.lower().endswith('.pdf'):
                    
                    continue

                # Request the page content from each URL
                async with session.get(url) as page_response:
                    # Try different encodings
                    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
                    page_content = None
                    
                    for encoding in encodings:
                        try:
                            page_content = await page_response.text(encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if page_content is None:
                        
                        continue

                    # Parse the HTML content using BeautifulSoup
                    soup = BeautifulSoup(page_content, 'html.parser')

                    # Search for email patterns in the page content
                    email_matches = re.findall(r'[\w\.-]+@[\w\.-]+', soup.text)
                    
                    if len(email_matches) > 1:
                        # Filter email matches based on author name
                        filtered_emails = []
                        author_parts = author_name.lower().split(" ")
                        for email in email_matches:
                            email_parts = email.split('@')[0].lower()
                            if any(part in email_parts for part in author_parts):
                                filtered_emails.append(email)
                        
                        
                    else:
                        filtered_emails = email_matches
                    # Return the first valid filtered email found
                    if filtered_emails:
                        return filtered_emails[0]


                    

        # If no emails were found
        print("No emails found.")
        return None

    except Exception as e:
        print(f"Error finding email for {author_name}: {str(e)}")
        return None



async def process_paper(paper, session):
    domain_task = classify_paper(paper['abstract'])
    summary_task = summarize_paper(paper['abstract'])
    impact_task = classify_impact(paper)
    
    domain, summary, impact = await asyncio.gather(domain_task, summary_task, impact_task)
    
    # Check if the paper is not AI-related
    domain_value = domain.split(': ')[1].strip() if ': ' in domain else domain.strip()
    if domain_value == "Not AI":
        return None  # Return None for non-AI papers
    
    # Handle potential variations in the impact string format
    impact_parts = impact.split('\n', 1)
    impact_classification = impact_parts[0] if impact_parts else ''
    impact_reason = impact_parts[1] if len(impact_parts) > 1 else ''
    
    impact_level = impact_classification.split(': ')[1] if ': ' in impact_classification else ''
    impact_reason = impact_reason.split(': ')[1] if ': ' in impact_reason else impact_reason.strip()
    
    # Get emails for authors
    author_emails = []
    email_tasks = [get_author_email(author, session) for author in paper['authors'].split(', ')]
    emails = await asyncio.gather(*email_tasks)
    
    for author, email in zip(paper['authors'].split(', '), emails):
        if email:
            author_emails.append(f"{author}: {email}")
    
    author_emails_str = '; '.join(author_emails)

    return [
        paper['title'],
        paper['published'].strftime('%Y-%m-%d'),
        paper['url'],
        domain_value,
        summary,
        paper['authors'],
        impact_level,
        author_emails_str,  # Add this new field
    ]

async def main():
    print("Starting main function")  # Debug print
    papers = get_arxiv_papers()
    
    # Check if CSV exists and read existing papers
    existing_papers = set()
    if os.path.exists('arxiv_papers.csv'):
        with open('arxiv_papers.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header
            for row in reader:
                if len(row) >= 3:  # Ensure the row has at least 3 columns
                    existing_papers.add(row[2])
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_paper(paper, session) for paper in papers if paper['url'] not in existing_papers]
        processed_papers = await asyncio.gather(*tasks)
    
    # Filter out None values (non-AI papers)
    processed_papers = [paper for paper in processed_papers if paper is not None]
    
    # Update Google Sheets
    # if processed_papers:
    #     update_google_sheets(processed_papers)
    
    # Append to local CSV
    with open('arxiv_papers.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # File is empty, write header
            writer.writerow(['Title', 'Published Date', 'URL', 'Domain', 'Impact Level', 'Summary', 'Authors', 'Author Emails'])
        writer.writerows(processed_papers)

if __name__ == "__main__":
    asyncio.run(main())