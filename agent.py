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
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from google.colab import userdata 

load_dotenv()

genai.configure(api_key=userdata.get('GOOGLE_API_KEY')) # type: ignore
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY')) # type: ignore


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

    images = [desktop_start,desktop_mid,mobile_start,mobile_mid]
    question = 'Evaluate these website screenshots and describe the user experience, design quality, and content organization. For the mobile screenshots note if the scaling or designing is off.'

    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": images,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)

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
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "system", "content": rating_prompt}
        ]
    )
    return completion.choices[0].message

evaluate_page('https://huggingface.co/deepseek-ai/Janus-Pro-7B')


    
        
            
    
    
