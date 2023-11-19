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
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import dotenv
import os

MONGODB_URI = "mongodb+srv://backupofamrit:GrJDmcTLkqxnR7Bo@aanlysiscluster.vwrt8og.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGODB_URI)
db = client["user_info"]
collection = db["addressconsole"]
dataset = load_dataset("Amod/mental_health_counseling_conversations")
corpus = [example['Context'] for example in dataset['train']]
dotenv.load_dotenv()
openai.api_key = os.environ['chatgpt_key']





model = whisper.load_model('base')
result = model.transcribe("/Users/siddsatish/Desktop/finer_vid.mov", fp16=False)

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

def analyze_sentiment(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or another GPT-4 model
        prompt=f"Analyze the following text to determine the person's emotional state and possible reasons for these emotions?"
               f" Please write the response as you are talking to the person directly and give them an analysis of their emotional state"
               f". Make sure the response is 3-4 sentences.\n\nText: \"{text}\"",
        max_tokens=130
    )
    return response.choices[0].text.strip()

sentiment = analyze_sentiment(preprocessed_text)

cap = cv2.VideoCapture("/Users/siddsatish/Desktop/finer_vid.mov")
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

# Insert the document into MongoDB
result = collection.insert_one(document)

plt.figure(figsize=(10, 6))
plt.bar(emotions, counts, color='skyblue')
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.title('Emotional States Distribution')
plt.xticks(rotation=45)
plt.savefig('emotion_graph.png', bbox_inches='tight')


blob = TextBlob(preprocessed_text)

sentiment_polarity = blob.sentiment.polarity
sentiment_subjectivity = blob.sentiment.subjectivity

introduction_text = """
This Analysis Report provides a comprehensive overview of the emotional states 
expressed through both facial expressions and speech, as observed in the provided video footage. The report combines advanced AI-driven analysis techniques, including sentiment analysis of transcribed speech, keyword significance evaluation, and facial emotion recognition.

The following contents are included in this report:

1. Summary of Sentiment Analysis: Highlights the overall emotional tone and subjectivity of the transcribed speech.
 
2. Significant Keywords: Identifies key terms that stand out in the speech, particularly relevant to mental health discussions.
 
3. Facial Emotion Analysis: Presents a distribution of emotional states detected through facial analysis.
 
4. Comparative Insights: Offers an integrated view by comparing findings from both speech and facial analysis.

Please note that this report is generated using AI algorithms and should be used as a supplementary tool for understanding emotional expressions. 
"""

pdf_file = "Analysis_Report.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter

# Create a text object
text_object = c.beginText()
text_object.setTextOrigin(50, height)
text_object.setFont("Helvetica", 9)

# Define the width for the text (adjust as needed)
max_width = 500

def add_wrapped_text(canvas, text_object, text, max_width, start_y, page_height):
    lines = text.split('\n')
    y_position = start_y
    line_height = 14  # Adjust as needed

    for line in lines:
        words = line.split()
        wrapped_line = ''
        for word in words:
            # Check if adding the word will exceed the line width
            if canvas.stringWidth(wrapped_line + word, "Helvetica", 9) < max_width:
                wrapped_line += word + ' '
            else:
                # If the line's width is exceeded, draw the line and start a new one
                text_object.setTextOrigin(50, y_position)
                text_object.textLine(wrapped_line)
                y_position -= line_height
                wrapped_line = word + ' '

                # Check for new page
                if y_position < 50:  # Adjust the bottom margin as needed
                    y_position = page_height - 50
                    canvas.drawText(text_object)
                    canvas.showPage()
                    text_object = canvas.beginText()
                    text_object.setFont("Helvetica", 9)
                    text_object.setTextOrigin(50, y_position)

        # Draw the last line
        text_object.setTextOrigin(50, y_position)
        text_object.textLine(wrapped_line)
        y_position -= line_height

    canvas.drawText(text_object)

# Usage
text_object = c.beginText()
text_object.setFont("Helvetica", 9)
add_wrapped_text(c, text_object, introduction_text, max_width, height-25, height)
text_object = c.beginText()
text_object.setFont("Helvetica-Bold", 12)
add_wrapped_text(c,text_object,"Session Overview:",max_width,height-280,height)
text_object = c.beginText()
text_object.setFont("Helvetica",9)
add_wrapped_text(c,text_object,sentiment,max_width,height-300,height)

dominant_emotion = max(emotions_dict, key=emotions_dict.get)
dominant_emotion_count = emotions_dict[dominant_emotion]

dominant_emotion_message = (f"The dominant emotion expressed in the video is '{dominant_emotion}' "
                            f"with a count of {dominant_emotion_count}. This suggests that during the video,"
                            f" the most frequently observed emotional expression was one of '{dominant_emotion}'.")

text_object = c.beginText()
text_object.setFont("Helvetica-Bold",12)
add_wrapped_text(c,text_object,"Emotional Status Review",max_width,height-350,height)

text_object = c.beginText()
text_object.setFont("Helvetica",9)
add_wrapped_text(c,text_object,dominant_emotion_message,max_width,height-370,height)

image_width = 200  # Adjust the width as needed
image_height = 100  # Adjust the height as needed

# Calculate the position to center the image (optional)
image_x = (width - image_width) / 2  # Center the image horizontally
image_y = height - 620  # Adjust the vertical position as needed

# Draw the image with the specified size
c.drawImage('emotion_graph.png', 50, image_y, width=image_width, height=image_height)


print(sentiment)
c.save()
