import google.generativeai as genai
import os

# --- 1. CONFIGURE GEMINI ---
# Put your API key here (in a real app, use environment variables!)
genai.configure(api_key="AIzaSyDZuZaIIA-dKcUqEt2j0XmbEO0NJMig0r0")

# Use the multimodal Flash model for speed
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_tactical_report(before_img, after_img, mask_img, lat, lon):
    print("🧠 Initiating NLP Tactical Analysis...")
    
    prompt = f"""
    You are an expert geospatial intelligence analyst. 
    I am providing you with three images of coordinates {lat}, {lon}:
    1. A 'Before' satellite image.
    2. An 'After' satellite image.
    3. An AI-generated change detection mask (white/colored pixels indicate physical changes on the ground).
    
    Write a concise, highly professional 3-sentence tactical report summarizing the geographical or structural changes detected. Be direct and objective.
    """
    
    # Feed the images and the prompt to Gemini
    response = model.generate_content([prompt, before_img, after_img, mask_img])
    
    return response.text