import google.generativeai as genai
import os 
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from transformers import AutoModel
from langchain_community.embeddings import OpenAIEmbeddings
import json
import torch
from PIL import Image
from lib.utlity import scrape_page,capture_desktop_and_mobile_screenshots
import torch
# from google import genai


load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY')) # type: ignore
# genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
openai_client= OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # type: ignore


def research_evaluation(query):
    model = genai.GenerativeModel('models/gemini-1.5-pro-002')
    response = model.generate_content(contents=query,
                                tools='google_search_retrieval')
    return response.candidates[0].content.parts[0].text


def evaluate_page(website):
    #pass in screenshot on desktop, mobile, then send scraped page
    # scraped_page = scrape_page(website)
    mobile_desktop_screenshots = capture_desktop_and_mobile_screenshots(website)
    desktop_start, desktop_mid = mobile_desktop_screenshots['desktop']
    mobile_start, mobile_mid = mobile_desktop_screenshots['mobile']

    prompt = 'Evaluate these website screenshots and describe the user experience, design quality, and content organization. For the mobile screenshots note if the scaling or designing is off. ignore any cookie banners. '
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    response = model.generate_content([desktop_start,desktop_mid,mobile_start,mobile_mid,prompt])
   
    return response
  

def page_quality_rating(website):
    #first try with just 1 reasoning layer, plugging in scraped website text and research from google ground api
    #Then try plugging in relevant context from manual, using the first reasoning layer to come up with relevant queries and using those to extract relevant text. That relevant text would then be sent to 2nd reasoning layer alongside research text and website text.
    #OR use website research results to extract most relevant chunks from vector db
    #OR simply extract relevant chunks based on YMYL, EEAT, Lowest, Low, Medium, High, Fully Meets and use those chunks
    #However website experience still needs to be judged, a seprate evaluation on the websites design (need image model llm), ads, mobile experience, and content,
    research_prompt = f'Look into {website} trusworthiness and authority. Then give me a score for their EEAT (Experience,Expertise, Authiritativeness, Trust) as well and indication of whether they are YMYL (Your money your life) from clearly not YMYL, possible YMYL, likely YMYL, to clearly YMYL. Make sure to visit trustpilot as well when determining this.'
    # website_text = scrape_page(website)
    website_research = research_evaluation(research_prompt)

    rating_prompt = f"""I'm going to give you text from a website as well as some research from it. From this you will give me a page quality rating according to US Rater guidelines (From lowest to fully meets)
            Website : {website}
            Website Research : {website_research}
            """
    
    #replace with deepseek R1 when available
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "system", "content": rating_prompt}
        ]
    )
    return completion.choices[0].message

print(evaluate_page('https://stackoverflow.com/questions/51046454/how-can-we-use-selenium-webdriver-in-colab-research-google-com'))


    
        
            
    
    
