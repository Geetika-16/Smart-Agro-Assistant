# 🌱 Smart Agro Assistant

## Description

Smart Agro Assistant is an AI-powered multilingual agriculture support system designed to assist farmers in plant disease detection and crop recommendation. The system provides support in 152 languages, allowing farmers to interact with the platform using both text and voice. It combines Machine Learning, Natural Language Processing, Speech Technologies, and Large Language Models to deliver accurate agricultural guidance in the user's preferred language.

The platform aims to make modern agricultural technologies more accessible to farmers by offering disease diagnosis, treatment recommendations, crop information, voice assistance, and report management through a simple web-based interface.

---

## Features

### 🌐 Multilingual Support

* Supports 152 languages.
* User can select their preferred language.
* Website content is automatically translated into the selected language.
* Supports both text and voice-based interaction.

### 🔊 Voice Assistance

* Speech-to-Text (STT) for voice input.
* Text-to-Speech (TTS) for reading results aloud.
* Helps farmers who may have difficulty reading text.

### 👤 User Authentication

* User registration and login system.
* Registration using Name, Phone Number, and Date of Birth.
* Easy login using Phone Number.
* User details securely stored in the database.

### 🍃 Plant Disease Detection

* Upload or capture plant leaf images.
* CNN model analyzes the uploaded image.
* Predicts plant disease name.
* Displays confidence score.
* AI-powered treatment generation.
* Provides both organic and pesticide-based treatment recommendations.
* Results can be read aloud in the selected language.
* Disease reports can be saved for future reference.

### 🌾 Crop Recommendation System

* Crop information retrieved from an integrated dataset.
* Dataset contains over 400 crops.
* Displays:

  * Crop Name
  * Crop Category
  * Soil Requirements
  * Water Requirements
  * Sunlight Requirements
  * Suitable Climate
  * Harvesting Time
* Results available in text and voice format.
* Crop reports can be saved.

### 📄 Saved Reports

* Users can view previously saved reports.
* Supports both disease detection reports and crop recommendation reports.
* Reports are displayed in the language in which they were saved.
* Read-aloud functionality available for saved reports.

---

## Technologies Used

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask (Python)

### Database

* MySQL

### Machine Learning & AI

* TensorFlow
* Convolutional Neural Network (CNN)
* Hugging Face LLM (Mistral)

### NLP & Language Processing

* Google Translator API
* Language Translation Module

### Voice Technologies

* Speech-to-Text (STT)
* Text-to-Speech (TTS)
* gTTS

### Python Libraries

* TensorFlow
* NumPy
* Pandas
* OpenCV
* Flask
* gTTS
* SpeechRecognition

---

## Project Workflow

### Step 1: Language Selection

1. User opens the Smart Agro Assistant website.
2. User selects a preferred language from 152 available languages.
3. Website content is automatically translated into the selected language.

### Step 2: User Authentication

1. New users register using:

   * Name
   * Phone Number
   * Date of Birth
2. User details are stored in the database.
3. Existing users log in using their phone number.
4. User is redirected to the dashboard.

### Step 3: Dashboard Access

The dashboard provides access to:

* Plant Disease Detection
* Crop Recommendation
* Saved Reports
* Support Services

### Step 4: Plant Disease Detection

1. User uploads or captures a plant leaf image.
2. Image is submitted for analysis.
3. CNN model processes the image.
4. Disease name is predicted.
5. Confidence score is generated.
6. Mistral LLM generates:

   * Organic treatments
   * Pesticide-based treatments
7. Results are displayed in the selected language.
8. User can listen to results using the read-aloud feature.
9. Report can be saved to the database.

### Step 5: Crop Recommendation

1. User searches for a crop.
2. System retrieves crop details from the integrated dataset.
3. Information displayed includes:

   * Crop Category
   * Soil Type
   * Water Requirements
   * Sunlight Requirements
   * Climate Conditions
   * Harvesting Time
4. Information is translated into the selected language.
5. Read-aloud feature provides voice output.
6. Report can be saved.

### Step 6: Saved Reports

1. User accesses the Saved Reports section.
2. Previously saved disease and crop reports are retrieved from the database.
3. Reports are displayed in their original saved language.
4. User can review information anytime.
5. Read-aloud option is available for all saved reports.

---

## Outcome

Smart Agro Assistant provides an intelligent, multilingual, and farmer-friendly platform that combines Artificial Intelligence, Machine Learning, Voice Technologies, and Agricultural Knowledge to support farmers in making informed decisions regarding plant health and crop management.

# Accuracy and Loss
<img width="1200" height="500" alt="accuracy_loss_plot" src="https://github.com/user-attachments/assets/650a4efc-fd20-49d9-858b-69f0d6465d91" />

# Confusion Matrix
<img width="1000" height="800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c82a9c4a-fde0-40f7-b58f-b911586b1968" />

