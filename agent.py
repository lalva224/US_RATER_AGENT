import google.generativeai as genai
import os 
import requests
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def searching_unit(query):
    model = genai.GenerativeModel('models/gemini-1.5-pro-002')
    response = model.generate_content(contents=query,
                                tools='google_search_retrieval')
    return response.candidates[0].content.parts[0].text

def scrape_page(website):
    url = f'https://r.jina.ai/{website}'
    headers = {'Authorization': f"Bearer {os.getenv('JINA_API_KEY')}"}

    response = requests.get(url, headers=headers)

    return response.text

def generate_guideline_relevant_queries(website):
    #first try with just 1 reasoning layer, plugging in scraped website text and research from google ground api
    #Then try plugging in relevant context from manual, using the first reasoning layer to come up with relevant queries and using those to extract relevant text. That relevant text would then be sent to 2nd reasoning layer alongside research text and website text.
    research_prompt = f'Look into {website} trusworthiness and authority. Then give me a score for their EEAT (Experience,Expertise, Authiritativeness, Trust) as well and indication of whether they are YMYL (Your money your life) from clearly not YMYL, possible YMYL, likely YMYL, to clearly YMYL. Make sure to visit trustpilot as well when determining this.'
    # website_text = scrape_page(website)
    website_research = searching_unit(research_prompt)

    rating_prompt = f"""I'm going to give you text from a website as well as some research from it. From this you will give me a page quality rating according to US Rater guidelines (From lowest to fully meets)
            Website : {website}
            Website Research : {website_research}
            """
    
    #replace with deepseek R1 when available
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "system", "content": rating_prompt}
        ]
    )
    return completion.choices[0].message

print(generate_guideline_relevant_queries('https://www.couponxoo.com/discount-nikon-scopes'))
# print(scrape_page('https://www.couponxoo.com/discount-nikon-scopes'))




    
        
            
    
    
