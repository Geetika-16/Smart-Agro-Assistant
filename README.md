# Smart-Agro-Assistant
Smart Agro Assistant is a multilingual AI tool that helps farmers with plant disease detection and crop recommendations. It uses image-based CNN models, AI-generated treatment advice, and voice/text interaction to make farming smarter and more accessible, even for non-technical users.

# Features

1. Multilingual input (speech and text)
2. Language selection
3. Selection of Plant Disesase Detection or Crop Recommendation or Both
4. Plant disease detection using image (CNN model)
5. AI-generated solutions (organic & chemical)
6. Crop recommendation based on soil, water, climate
7. Speech output using gTTS
8. Offline-capable AI chatbot (using LLaMA)
9. Easy-to-use voice interface for non-technical users

# Tools

1. Python 3.10 version
2. Libraries- TensorFlow, OpenCV, NumPy, Pandas, gTTS, SpeechRecognition, Googletrans.
3. AI Model- CNN for disease detection, LLaMA (offline) for AI responses 
4. Voice- gTTS (Text-to-Speech), SpeechRecognition (Speech-to-Text) 
5. Language Detection- Google Translate API 

# Installation

1. **Clone the repository**
   git clone https://github.com/Geetika-16/Smart-Agro-Assistant.git
   cd Smart-Agro-Assistant
   
2. **Setup Environment**
    .\cnn_env\Scripts\activate (Runs Under the CNN environment)
    python "Main.py"  (File name)

3. **Install Dependencies**
    pip install required libraries (Note: Tensorflow and Keras should be in same version Eg: tensorflow==2.13.0 and Keras==2.13.1)
