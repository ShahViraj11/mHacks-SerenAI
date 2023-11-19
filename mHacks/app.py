import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv, find_dotenv
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_client import OAuth
from flask import Flask, redirect, render_template, session, url_for, request, send_from_directory
from google.cloud import storage
import dateutil.parser


uri = "mongodb+srv://backupofamrit:GrJDmcTLkqxnR7Bo@aanlysiscluster.vwrt8og.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\thelo\OneDrive\Desktop\Python Projects\mHacks-SerenAI\mHacks\serenai-405517-7d5605a5a880.json"
app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")

oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)

def dispdf(user_db):
    emotions = list(user_db['emotions_dict'].keys())
    counts = list(user_db['emotions_dict'].values())

    plt.figure(figsize=(10, 6))
    plt.bar(emotions, counts, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Emotional States Distribution')
    plt.xticks(rotation=45)
    plt.savefig('emotion_graph.png', bbox_inches='tight')


    introduction_text = """
    This Analysis Report provides a comprehensive overview of the emotional states expressed through both facial expressions and speech, as observed in the provided video footage. The report combines advanced AI-driven analysis techniques, including sentiment analysis of transcribed speech, keyword significance evaluation, and facial emotion recognition.

    The following contents are included in this report:

    1. Summary of Sentiment Analysis: Highlights the overall emotional tone and subjectivity of the transcribed speech.
    2. Significant Keywords: Identifies key terms that stand out in the speech, particularly relevant to mental health discussions.
    3. Facial Emotion Analysis: Presents a distribution of emotional states detected through facial analysis.
    4. Comparative Insights: Offers an integrated view by comparing findings from both speech and facial analysis.

    Please note that this report is generated using AI algorithms and should be used as a supplementary tool for understanding emotional expressions.
    """

    pdf_file = "mHacks/static/Analysis_Report.pdf"

    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter

    title = "Analysis Report"
    title_font_size = 18
    title_text_object = c.beginText()
    title_text_object.setFont("Helvetica-Bold", title_font_size)
    title_text_object.setTextOrigin((width - c.stringWidth(title, "Helvetica-Bold", title_font_size)) / 2, height - 50)
    title_text_object.textLine(title)
    c.drawText(title_text_object)
    # Create a text object
    text_object = c.beginText()
    text_object.setTextOrigin(50, height-55)
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
    add_wrapped_text(c, text_object, introduction_text, max_width, height - 55, height)
    text_object = c.beginText()
    text_object.setFont("Helvetica-Bold", 12)
    add_wrapped_text(c, text_object, "Session Overview:", max_width, height - 270, height)
    text_object = c.beginText()
    text_object.setFont("Helvetica", 9)
    add_wrapped_text(c, text_object, user_db['sentiment'], max_width, height - 290, height)


    text_object = c.beginText()
    text_object.setFont("Helvetica-Bold", 12)
    add_wrapped_text(c, text_object, "Emotional Status Review", max_width, height - 370, height)

    text_object = c.beginText()
    text_object.setFont("Helvetica", 9)
    add_wrapped_text(c, text_object, user_db['dominant_emotion_message'], max_width, height - 390, height)

    image_width = 165  # Adjust the width as needed
    image_height = 135  # Adjust the height as needed

    # Calculate the position to center the image (optional)
    image_x = (width - image_width) / 2  # Center the image horizontally
    image_y = height - 575  # Adjust the vertical position as needed

    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    plt.pie(user_db['scores'], labels=user_db['words'], autopct='%1.1f%%', startangle=140)
    plt.title('Top 5 Influential Words')
    plt.savefig('pie_chart.png', bbox_inches='tight')

    # Draw the image with the specified size
    c.drawImage('emotion_graph.png', 50, image_y, width=image_width, height=image_height)
    c.drawImage('pie_chart.png',290, image_y, width=image_width, height=image_height)

    text_object = c.beginText()
    text_object.setFont("Helvetica", 9)
    add_wrapped_text(c, text_object, user_db['sent_analysis'], max_width, height - 600, height)

    print(pdf_file)
    c.save()

    return "working"


@app.route("/")
def home():
    return render_template(
        "index.html",
        session=session.get("user"),
        pretty=json.dumps(session.get("user"), indent=4),
    )


@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://"
        + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("index", _external=True),
                "client_id": env.get("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )


@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    global info_dict
    info_dict = dict(session)
    session["user"] = token
    return redirect("/form")  # Redirect to the form page


@app.route("/form")
def form():
    if 'user' in session:
        return render_template("secondpage.html")
    else:
        return redirect("/")


@app.route('/submit_form', methods=['POST'])
def submit_form():
    uploaded_file = request.files['video_upload']

    if uploaded_file:
        storage_client = storage.Client()
        bucket_name = 'user_mood_bucket'
        destination_blob_name = 'user_video.mp4'
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(uploaded_file)
        blob.content_type = 'video/mp4'
        blob.patch()
    

    form_data = request.form
    date_of_complaint = form_data.get('date_of_complaint')
    first_name = form_data.get('first_name')
    last_name = form_data.get('last_name')
    street_address = form_data.get('street_address')
    street_address2 = form_data.get('street_address2')
    city = form_data.get('city')
    region = form_data.get('region')
    postal_code = form_data.get('postal_code')
    country = form_data.get('country')
    incident_location = form_data.get('incident_location')
    consult_details = form_data.get('consult_details')
    signature = form_data.get('signature')

    db = client['user_info']
    coll = db["addressconsole"]

    
    item = {
        "_id": info_dict['user']['userinfo']['email'],
        "date_of_complaint": date_of_complaint,
        "name": {"first": first_name, "last": last_name},
        "address": {
            "street": street_address,
            "street2": street_address2,
            "city": city,
            "region": region,
            "postal_code": postal_code,
            "country": country
        },
        "incident_location": incident_location,
        "consult_details": consult_details,
        "signature": signature
    }

    coll.insert_one(item)
    return render_template("thank_you.html")

@app.route('/pdf_redirect', methods=['POST'])
def pdf_check():
    db = client['user_info']
    coll = db['videoanalyses']
    data = coll.find()
    sorted_data = sorted(data, key=lambda x: dateutil.parser.parse(x['updated']), reverse=True)
    dispdf(sorted_data[0])

    return send_from_directory('C:\\Users\\thelo\\OneDrive\\Desktop\\Python Projects\\mHacks-SerenAI\\mHacks\\static',
                               'Analysis_Report.pdf', as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))
 