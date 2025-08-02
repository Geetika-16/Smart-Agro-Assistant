import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from googletrans import Translator
from tensorflow.keras.preprocessing import image
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from llama_cpp import Llama


# ============== Global Settings ==============
read_aloud = True
language_code = 'en'
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
    prompt = f"""You are an agriculture expert.
A farmer has a crop suffering from this disease: {disease_name}.
Please provide:

1. Organic Treatment Methods
2. Pesticide-Based Treatment Methods

Your answer should be detailed, informative, and relevant to Indian agricultural practices."""

    # Call the model
    output = llm(prompt, max_tokens=512, stop=["</s>"])
    
    # Parse and return only the response text
    response_text = output["choices"][0]["text"].strip()

    print("[AI Response]:", response_text)
    return response_text




# ============== Voice Input ==============
def get_audio():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("[System]: Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source)
            print("[System]: Listening...")
            audio = r.listen(source, timeout=10)
            text = r.recognize_google(audio)
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
def speak(text, lang='en'):
    if not text.strip():
        print("[System]: (No text to speak)")
        return
    print(f"[System]: {text}")
    if read_aloud:
        try:
            tts = gTTS(text=text, lang=lang)
            output_path = "D:\\Desktop\\Project\\voice_output1.mp3"
            tts.save(output_path)
            playsound(output_path)
            os.remove(output_path)
        except Exception as e:
            print(f"[TTS Error]: {e}")

# ============== Language Detection ==============
# Load the language CSV into a dictionary (once at top)
language_df = pd.read_csv("Language_Dataset.csv")  # Columns: Language, Language Code
# Replace your existing language_map line with:
language_map = {
    str(row['Language']).strip().lower(): str(row['Language Code']).strip()
    for _, row in language_df.iterrows()
    if pd.notna(row['Language']) and pd.notna(row['Language Code'])
}
# ============== Helper: Detect Language Code ==============
def detect_language_code(user_input_language):
    user_input_language = str(user_input_language).strip().lower()
    return language_map.get(user_input_language, 'en')  # fallback to English



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
    index = np.argmax(prediction)
    return class_labels[index]

# ============== Crop Info Lookup ==============
def get_crop_info(crop_name, df):
    row = df[df['Crop_Name'].str.lower() == crop_name.lower()]
    if row.empty:
        return None
    return {
        "Water": row.iloc[0]["Water_(L/season)"],
        "Sunlight": row.iloc[0]["Sunlight"],
        "Soil_type": row.iloc[0]["Soil_Type"],
        "Climate": row.iloc[0].get("Best_Climate", "Not available")
    }

# ============== Main Program ==============
def main():
    global read_aloud, language_code

    # Load models and data
    model = tf.keras.models.load_model(r"CNN_Model.h5")
    with open(r"class_names_new.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    crop_df = pd.read_csv(r"Crop_dataset_cleaned.csv")

    # --- Language selection ---
    print("[System]: Enter or say your language (e.g., English, Tamil, Hindi):")
    lang_input = input("Language or press Enter to speak: ").strip()
    if not lang_input:
        speak("Please say the language now.", "en")
        lang_input = get_audio()
    language_code = detect_language_code(lang_input)

    # --- Read aloud preference ---
    speak("Do you want me to read the information aloud? Say yes or no.", language_code)
    tts_pref = input("Read aloud? (yes/no): ").strip()
    if not tts_pref:
        speak("Please say yes or no.", language_code)
        tts_pref = get_audio()
    read_aloud = "yes" in tts_pref.lower()

    # --- Service selection ---
    speak("Do you want plant disease detection, crop recommendation, or both?", language_code)
    choice = input("Your choice (or press Enter to say): ").strip().lower()
    if not choice:
        speak("Please say your choice now.", language_code)
        choice = get_audio().lower()

    # --- Disease detection ---
    if "disease" in choice or "both" in choice:
        speak("Please enter the image path of the plant.", language_code)
        img_path = input("Image path: ").strip()
        disease = classify_disease(img_path, model, class_labels)
        speak(f"Disease detected: {disease}", language_code)

        #  Skips AI if healthy
    if "healthy" in disease.lower():
        speak("The plant appears to be healthy. No treatment is required.", language_code)
    else:
        solution = get_local_ai_solution(disease)
        if not solution.strip():
            speak("The AI could not find a proper treatment for this disease. Please try again later.", language_code)
        else:
            print(f"\n Disease Detected: {disease}\n AI says:\nYou can treat {disease} using:\n")
            
            lines = solution.strip().split("\n")
            organic_part = []
            pesticide_part = []
            current = None
            
            for line in lines:
                line_lower = line.lower()
                if "organic treatment" in line_lower:
                    current = "organic"
                elif "pesticide treatment" in line_lower or "chemical treatment" in line_lower:
                    current = "pesticide"
                elif current == "organic":
                    organic_part.append(line)
                elif current == "pesticide":
                    pesticide_part.append(line)

            organic_text = "\n".join(organic_part).strip()
            pesticide_text = "\n".join(pesticide_part).strip()
                
            translated_organic = translator.translate(organic_text or "No organic treatment found.", dest=language_code).text
            translated_pesticide = translator.translate(pesticide_text or "No pesticide treatment found.", dest=language_code).text

                
            print("Organic Treatment:")
            print(translated_organic)
            print("\n Pesticide Treatment:")
            print(translated_pesticide)

            speak("Here are the organic treatment methods:", language_code)
            speak(translated_organic, language_code)
            
            speak("Here are the pesticide-based treatment methods:", language_code)
            speak(translated_pesticide, language_code)


    # --- Crop recommendation ---
    if "crop" in choice or "both" in choice:
        speak("Enter or say the crop name.", language_code)
        crop_name = input("Crop name: ").strip()
        if not crop_name:
            speak("Please say the crop name now.", language_code)
            crop_name = get_audio()
        crop_info = get_crop_info(crop_name, crop_df)
        if crop_info:
            speak(f"Water requirement: {crop_info['Water']}", language_code)
            speak(f"Sunlight needed: {crop_info['Sunlight']}", language_code)
            speak(f"Soil type: {crop_info['Soil_type']}", language_code)
            speak(f"Climate: {crop_info['Climate']}", language_code)
        else:
            speak("Crop not found in the dataset.", language_code)

    # --- Final confirmation ---
    speak("Would you like me to read the details again? Say yes or no.", language_code)
    final_pref = input("Read again? (yes/no): ").strip()
    if not final_pref:
        speak("Please say yes or no.", language_code)
        final_pref = get_audio()
    if "yes" in final_pref.lower():
        speak("You can rerun the program and enter your query again. Goodbye!", language_code)
    else:
        speak("Thank you for using Our Agro System. Goodbye!", language_code)

if __name__ == "__main__":
    main()
