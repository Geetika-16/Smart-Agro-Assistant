import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from llama_cpp import Llama
import re

# ==========================================

# LOAD CNN MODEL

# ==========================================

MODEL = tf.keras.models.load_model("CNN_Model.h5")

with open("class_names_new.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# LOAD MISTRAL MODEL

LLM = Llama(
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

# LOAD CROP DATASET
CROP_DF = pd.read_csv("Crop_dataset_cleaned.csv")
print(CROP_DF.head())
print(CROP_DF.columns)

# IMAGE PREPROCESSING
def preprocess_image(img_path):
    img = image.load_img(
    img_path,
    target_size=(128, 128)
)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(
    img_array,
    axis=0
)
    return img_array
# DISEASE PREDICTION
def predict_disease(img_path):
    img = preprocess_image(img_path)
    prediction = MODEL.predict(img)
    confidence = float(np.max(prediction))
    class_index = np.argmax(prediction)
    disease_name = CLASS_NAMES[class_index]
    formatted_name = disease_name.replace(
    "___",
    " - "
).replace(
    "_",
    " "
)
    return {
    "disease_name": formatted_name,
    "confidence": round(confidence, 4),
    "raw_class": disease_name
}

def parse_treatment(response):
    """Parse organic and chemical sections regardless of heading capitalisation."""
    organic = []
    chemical = []
 
    resp_upper = response.upper()
 
    # Find where ORGANIC block starts
    org_match = re.search(r'ORGANIC[S]?\s*:', resp_upper)
 
    # Find where PESTICIDE / CHEMICAL block starts (any variation)
    chem_match = re.search(r'(PESTICIDE[S]?|CHEMICAL[S]?)\s*:', resp_upper)
 
    if org_match and chem_match:
        org_text  = response[org_match.end() : chem_match.start()]
        chem_text = response[chem_match.end():]
    elif org_match:
        org_text  = response[org_match.end():]
        chem_text = ""
    else:
        # Fallback: treat whole response as organic if no headers found
        org_text  = response
        chem_text = ""
 
    # Only keep numbered lines (1. 2. 3. …)
    organic  = [l.strip() for l in org_text.split("\n")
                if l.strip() and l.strip()[0].isdigit()]
    chemical = [l.strip() for l in chem_text.split("\n")
                if l.strip() and l.strip()[0].isdigit()]
 
    return {
        "organic":  organic  if organic  else ["No organic treatment found"],
        "chemical": chemical if chemical else ["No pesticide treatment found"]
    }
 
# AI TREATMENT GENERATION
def get_mistral_solution(disease_name):

    if "healthy" in disease_name.lower():
        return {
            "organic": ["Plant is healthy.","No treatment required."],
            "chemical": ["No pesticide required."]
        }

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

Only provide treatments.

[/INST]
"""
    
    LLM.reset()
    output = LLM(
        prompt,
        max_tokens=400,
        stop=["</s>"]
    )

    response = output["choices"][0]["text"].strip()

    organic = []
    chemical = []

    if "PESTICIDE:" in response:

        parts = response.split("PESTICIDE:")

        organic_text = parts[0].replace(
            "ORGANIC:",
            ""
        ).strip()

        pesticide_text = parts[1].strip()

        organic = [
            line.strip()
            for line in organic_text.split("\n")
            if line.strip()
        ]

        chemical = [
            line.strip()
            for line in pesticide_text.split("\n")
            if line.strip()
        ]

    return {
        "organic": organic if organic else ["No organic treatment found"],
        "chemical": chemical if chemical else ["No pesticide treatment found"]
    }
        

# CROP INFORMATION
def get_crop_info(crop_name):
    
    crop_name = crop_name.lower().strip()
    crop_name = re.sub(r'[^a-zA-Z0-9\s]', '', crop_name)
    matches = CROP_DF[
    CROP_DF["Crop_Name"]
    .astype(str)
    .str.lower()
    .str.contains(
        crop_name,
        na=False
    )
]
    if matches.empty:
        return None
    
    row = matches.iloc[0]
    
    return {
        "Crop_Name": str(row["Crop_Name"]),
        "Category": str(row["Category"]),
        "Water": str(row["Water_(L/season)"]),
        "Sunlight": str(row["Sunlight"]),
        "Soil_Type": str(row["Soil_Type"]),
        "Best_Climate": str(row["Best_Climate"]),
        "Harvest_Time": str(row["Harvest_Time"])
}


# ==========================================

# COMPLETE DISEASE RESULT

# ==========================================

def analyze_disease(img_path):
    prediction = predict_disease(img_path)
    
    treatment = get_mistral_solution(
        prediction["disease_name"]
)
    return {
        "disease_name": prediction["disease_name"],
        "confidence": prediction["confidence"],
        "organic": treatment["organic"],
        "chemical": treatment["chemical"]
}
if __name__ == "__main__":

    result = analyze_disease("Test\Test 2.JPG")

    print(result)

    crop = get_crop_info("Tomato")

    print(crop)


