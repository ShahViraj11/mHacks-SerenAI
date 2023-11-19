import functions_framework
from google.cloud import storage
from pymongo import MongoClient
import whisper
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import openai
import cv2
from deepface import DeepFace
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


MONGODB_URI = "mongodb+srv://backupofamrit:GrJDmcTLkqxnR7Bo@aanlysiscluster.vwrt8og.mongodb.net/?retryWrites=true&w=majority"
openai.api_key = "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"

client = MongoClient(MONGODB_URI)
db = client["user_info"]
collection = db["addressconsole"]
dataset = load_dataset("Amod/mental_health_counseling_conversations")
corpus = [example['Context'] for example in dataset['train']]

def analyze_sentiment(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or another GPT-4 model
        prompt=f"Analyze the following text to determine the person's emotional state and possible reasons for these emotions?"
               f" Please write the response as you are talking to the person directly and give them an analysis of their emotional state"
               f". Make sure the response is 3-4 sentences.\n\nText: \"{text}\"",
        max_tokens=130
    )
    return response.choices[0].text.strip()

@functions_framework.cloud_event
def hello_gcs(cloud_event):
    storage_client = storage.Client()
    bucket_name = 'user_mood_bucket'
    video_file_name = 'finer_vid.mov'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(video_file_name)
    temp_video_path = '/tmp/temp_video.mov'
    blob.download_to_filename(temp_video_path)
    model = whisper.load_model('base')
    result = model.transcribe(temp_video_path, fp16=False)
    transcribed_text = result["text"]
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    tokens = word_tokenize(transcribed_text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

    preprocessed_text = ' '.join(lemmatized)
    corpus.append(transcribed_text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[-1]
    word_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}
    sorted_word_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    sentiment = analyze_sentiment(preprocessed_text)

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    emotions_dict = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for i in range(int(fps)-1):
            cap.read()  # Read and discard

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            try:
                emotions_dict[result[0]['dominant_emotion']] += 1
            except KeyError:
                emotions_dict[result[0]['dominant_emotion']] = 1
            print(result[0]['dominant_emotion'])
        except Exception as e:
            print("Error in emotion detection", e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    emotions = list(emotions_dict.keys())
    counts = list(emotions_dict.values())

    document = {
    "event_id": "test_video_id",
    "sentiment": sentiment,
    "dominant_emotion": max(emotions_dict, key=emotions_dict.get),
    }

    result = collection.insert_one(document)


if _name_ == "_main_":
    hello_gcs(None)  # For testing the function locally