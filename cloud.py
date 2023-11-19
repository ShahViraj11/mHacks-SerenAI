dataset = load_dataset("Amod/mental_health_counseling_conversations")
corpus = [example['Context'] for example in dataset['train']]

dotenv.load_dotenv()

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

def sentiment_analysis(sentiment_polarity, sentiment_subjectivity, top_words):
    top_words_str = ', '.join([f"{word}: {score:.2f}" for word, score in top_words[:10]])
    # Construct the prompt
    prompt = (
        f"This analysis is based on the sentiment scores and significant keywords from the text. "
        f"The sentiment polarity is {sentiment_polarity:.2f} and subjectivity is {sentiment_subjectivity:.2f}. "
        f"The top significant words are {top_words_str}. "
        f"Please analyze what these sentiment scores and keywords imply about the person's emotional state. "
        f"Provide a 4-5 sentence response discussing the emotional state and the relevance of these words."
        f"Also, Please write the response as you are talking to the person directly. "
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()


def analyze_sentiment(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or another GPT-4 model
        prompt=f"Analyze the following text to determine the person's emotional state and possible reasons for these emotions?"
               f" Please write the response as you are talking to the person directly and give them an analysis of their emotional state"
               f". Make sure the response is 3-4 sentences.\n\nText: \"{text}\"",
        max_tokens=150
    )
    return response.choices[0].text.strip()

sentiment = analyze_sentiment(preprocessed_text)

cap = cv2.VideoCapture("/Users/siddsatish/Desktop/finer_vid.mov")
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video

emotions_dict = {}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process one frame per second
    for i in range(int(fps) - 1):
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
print(emotions_dict)
cap.release()
cv2.destroyAllWindows()

blob = TextBlob(preprocessed_text)

# Get the sentiment polarity
sentiment_polarity = blob.sentiment.polarity

# Get the sentiment subjectivity
sentiment_subjectivity = blob.sentiment.subjectivity

sent_analysis = sentiment_analysis(sentiment_polarity, sentiment_subjectivity, sorted_word_scores)
####################

dominant_emotion = max(user_db['emotions_dict'], key=user_db['emotions_dict'].get)
dominant_emotion_count = user_db['emotions_dict'][dominant_emotion]

dominant_emotion_message = (f"The dominant emotion expressed in the video is '{dominant_emotion}' "
                            f"with a count of {dominant_emotion_count}. This suggests that during the video,"
                            f" the most frequently observed emotional expression was one of '{dominant_emotion}'.")
