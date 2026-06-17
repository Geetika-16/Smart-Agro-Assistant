import speech_recognition as sr
from gtts import gTTS
import pygame
import uuid
from googletrans import Translator
from tensorflow.keras.preprocessing import image
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from llama_cpp import Llama

pygame.mixer.init()
# ============== Global Settings ==============
read_aloud = True
translate_code = "en"
tts_code = "en"
speech_code = "en-US"
translator = Translator()
def safe_translate(text, dest_lang):
    if not text.strip():
        return text
    try:
        translated = translator.translate(text, dest=dest_lang).text
        return translated
    except Exception as e:
        print(f"[Translation Error]: {e}")
        return text  # fallback to original if translation fails


# ============== Load Offline AI Model (LLaMA) ==============
llm = Llama(
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,     # adjust based on your CPU
    n_gpu_layers=20, # optional: if you want GPU acceleration
    verbose=False
)

def get_local_ai_solution(disease_name):
    print(f"[System]: Generating AI response for disease: {disease_name}...")

    # Prepare a clear prompt with instruction format
    prompt = f"""
[INST]

You are a plant disease expert.

Disease: {disease_name}

Give:

ORGANIC:
1.
2.
3.

PESTICIDE:
1.
2.
3.

If you do not know the disease, write:

ORGANIC:
No treatment available.

PESTICIDE:
No treatment available.

Only give treatments.

[/INST]
"""

    # Call the model
    output = llm(
    prompt,
    max_tokens=512,
    stop=["</s>"]
)
    
    # Parse and return only the response text
    response_text = output["choices"][0]["text"].strip()

    print("\nRAW AI OUTPUT", response_text)   
    return response_text




# ============== Voice Input ==============

def get_audio():

    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    try:
        with sr.Microphone() as source:
            print("[System]: Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=2)
            
            print("[System]: Listening...")
            audio = r.listen(source, timeout=15, phrase_time_limit=12)
            text = r.recognize_google(audio, language=speech_code)
            print(f"[Voice Input]: {text}")
            return text.lower()
    except sr.WaitTimeoutError:
        print("[Voice Error]: Listening timed out.")
    except sr.UnknownValueError:
        print("[Voice Error]: Could not understand audio.")
    except sr.RequestError:
        print("[Voice Error]: Google API error.")
    except Exception as e:
        print(f"[Voice Error]: {e}")
    return ""

# ============== Text to Speech ==============
def speak(text):
    if not text.strip():
        return

    print(f"[System]: {text}")

    if read_aloud:
        try:
            filename = f"tts_{uuid.uuid4().hex}.mp3"

            tts = gTTS(text=text, lang=tts_code)
            tts.save(filename)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pass

            pygame.mixer.music.unload()
            os.remove(filename)

        except Exception as e:
            print("[TTS Error]:", e)

# ============== Language Detection ==============
# Load the language CSV into a dictionary (once at top)
language_df = pd.read_csv("Language_Dataset.csv")  # Columns: Language, Language Code
# Replace your existing language_map line with:English

translate_map = {}
tts_map = {}
speech_map = {}

for _, row in language_df.iterrows():

    language_name = str(row['Language']).strip().lower()

    translate_map[language_name] = str(row['Translate_code']).strip()
    tts_map[language_name] = str(row['TTS_Code']).strip()
    speech_map[language_name] = str(row['Speech_Code']).strip()
# ============== Helper: Detect Language Code ==============
def detect_language(user_input_language):
    user_input_language = str(user_input_language).strip().lower()

    return (
        translate_map.get(user_input_language, "en"),
        tts_map.get(user_input_language, "en"),
        speech_map.get(user_input_language, "en-US")
    )



# ============== Image Preprocessing ==============
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ============== Disease Detection ==============
def classify_disease(img_path, model, class_labels):
    img = preprocess_image(img_path)

    prediction = model.predict(img)

    confidence = float(np.max(prediction))
    index = np.argmax(prediction)
    
    return class_labels[index], confidence

# ============== Crop Info Lookup ==============
def get_crop_info(crop_name, df):
    crop_name = crop_name.lower().strip()

    matches = df[
        df['Crop_Name']
        .str.lower()
        .str.contains(crop_name, na=False)
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    return {
        "Water": row["Water_(L/season)"],
        "Sunlight": row["Sunlight"],
        "Soil_type": row["Soil_Type"],
        "Climate": row.get("Best_Climate", "Not available"),
        "Category": row.get("Category", "Not available"),
        "Harvest_Time": row.get("Harvest_Time", "Not available")
    }
# ============== Main Program ==============

def format_disease_name(disease):
    disease = disease.replace("___", " - ")
    disease = disease.replace("_", " ")

    # remove unnecessary dataset words
    disease = disease.replace("(including sour)", "")

    return disease.strip()

def translate_crop_term(term):

    translated = safe_translate(
        str(term),
        translate_code
    )

    return translated
def main():
    global read_aloud
    global translate_code
    global tts_code
    global speech_code

    summary_text = ""

    # Load models and data
    model = tf.keras.models.load_model(r"CNN_Model.h5")
    with open(r"class_names_new.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    crop_df = pd.read_csv(r"Crop_dataset_cleaned.csv")
    

    # --- Language selection ---
    print("====================================================")
    print("             SMART AGRO SYSTEM WELCOMES YOU         ")
    print("====================================================")

    print("[System]: Enter or say your language (e.g., English, Tamil, Hindi):")
    lang_input = input("Language or press Enter to speak: ").strip()
    if not lang_input:
        speak("Please say the language now.")
        lang_input = get_audio()
    translate_code, tts_code, speech_code = detect_language(lang_input)

    # --- Read aloud preference ---
    speak("Do you want me to read the information aloud? Say yes or no.")
    tts_pref = input("Read aloud? (yes/no): ").strip()
    if not tts_pref:
        speak("Please say yes or no.")
        tts_pref = get_audio()
        
    tts_pref = safe_translate(tts_pref, "en").lower()
    
    read_aloud = "yes" in tts_pref

    # --- Service selection ---
    speak("Do you want plant disease detection, crop recommendation, or both?")
    choice = input("Your choice (or press Enter to say): ").strip()
    
    if not choice:
        speak("Please say your choice now.")
        choice = get_audio().lower()
    choice = safe_translate(choice, "en").lower()

    # --- Disease detection ---
    disease = ""
    if "disease" in choice or "both" in choice:
        speak("Please enter the image path of the plant.")
        img_path = input("Image path: ").strip()
        disease, confidence = classify_disease(img_path, model, class_labels)
        display_disease = format_disease_name(disease)
        translated_disease = safe_translate(
            display_disease,
            translate_code
            )
        confidence_percent = round(confidence * 100, 2)
        print(f"[System]: Confidence Score = {confidence_percent}%")
        if confidence_percent < 70:
            speak("Warning: The prediction confidence is low. The result may not be accurate.")
        
        speak(
           f"Disease detected: {translated_disease}. "
           f"Confidence score is {confidence_percent} percent."
           )
        summary_text += f"\nDisease: {translated_disease}\n"
    
    if disease:
        
        if "healthy" in disease.lower():
            speak(
            "The plant appears to be healthy. No treatment is required."
        )
        
        else:
            solution = get_local_ai_solution(format_disease_name(disease))
            
            if not solution.strip():
                speak("The AI could not find a proper treatment for this disease. Please try again later.")
            
            else:
                print(f"\n Disease Detected: {disease}\n AI says:\nYou can treat {disease} using:\n")
            
            organic_text = ""
            pesticide_text = ""
            if "PESTICIDE:" in solution:
                parts = solution.split("PESTICIDE:")
                organic_text = (
                    parts[0]
                    .replace("ORGANIC:", "")
                    .strip()
                    )
                pesticide_text = parts[1].strip()
            
            else:
                organic_text = solution
                
            translated_organic = safe_translate(organic_text or "No organic treatment found.", translate_code)
            translated_pesticide = safe_translate(pesticide_text or "No pesticide treatment found.", translate_code)
            summary_text += f"\nOrganic Treatment:\n{translated_organic}\n"
            summary_text += f"\nPesticide Treatment:\n{translated_pesticide}\n"
             
            print("Organic Treatment:")
            print(translated_organic)
            print("\n Pesticide Treatment:")
            print(translated_pesticide)
            

            speak("Here are the organic treatment methods:")
            speak(translated_organic)
            
            speak("Here are the pesticide-based treatment methods:")
            speak(translated_pesticide)
            
  
    # --- Crop recommendation ---
    if "crop" in choice or "both" in choice:
        speak("Enter or say the crop name.")
        crop_name = input("Crop name: ").strip()
        
        if not crop_name:
            speak("Please say the crop name now.")
            crop_name = get_audio()
        crop_name_english = safe_translate(crop_name,"en")
        crop_info = get_crop_info(crop_name_english,crop_df)
        
        
        if crop_info:
            water = translate_crop_term(crop_info['Water'])
            sunlight = translate_crop_term(crop_info['Sunlight'])
            soil = translate_crop_term(crop_info['Soil_type'])
            climate = translate_crop_term(crop_info['Climate'])
            category = translate_crop_term(crop_info['Category'])
            harvest_time = translate_crop_term(crop_info['Harvest_Time'])

            speak(f"Water requirement: {water}")
            speak(f"Sunlight needed: {sunlight}")
            speak(f"Soil type: {soil}")
            speak(f"Climate: {climate}")
            speak(f"Category: {category}")
            speak(f"Harvest time: {harvest_time}")

            summary_text += f"\nCrop: {crop_name}"
            summary_text += f"\nWater Requirement: {water}"
            summary_text += f"\nSunlight: {sunlight}" 
            summary_text += f"\nSoil Type: {soil}"
            summary_text += f"\nClimate: {climate}"
            summary_text += f"\nCategory: {category}"
            summary_text += f"\nHarvest Time: {harvest_time}"
            
            
            
        else:
            speak("Crop not found in the dataset.")
            

    # --- Final confirmation ---
    speak("Would you like me to read the details again? Say yes or no.")
    final_pref = input("Read again? (yes/no): ").strip()
    if not final_pref:
        speak("Please say yes or no.")
        final_pref = get_audio()
    final_pref = safe_translate(final_pref, "en").lower()
    
    if "yes" in final_pref.lower():
        speak("Reading your complete report again.")
        speak(summary_text)

        speak(
        "Thank you for using Our Agro System. Goodbye!"
    )
        
    else:
        speak(
        "Thank you for using Our Agro System. Goodbye!"
    )
if __name__ == "__main__":
    main()
