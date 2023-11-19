"""Python Flask WebApp Auth0 integration example
"""

import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, request
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://backupofamrit:GrJDmcTLkqxnR7Bo@aanlysiscluster.vwrt8og.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

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


# Controllers API
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
    session["user"] = token
    return redirect("/form")  # Redirect to the form page


@app.route("/form")
def form():
    # Check if the user is logged in before showing the form
    if 'user' in session:
        return render_template("secondpage.html")
    else:
        return redirect("/")


@app.route('/submit_form', methods=['POST'])
def submit_form():
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
        "_id" : "testing123",
        "date_of_complaint": date_of_complaint,
        "name": {"first": first_name, "last": last_name},
        "address": [street_address,
                    street_address2,
                    city,
                    region,
                    postal_code,
                    country],
        "incident_location": incident_location,
        "consult_details": consult_details,
        "signature": signature
    }
    coll.insert_one(item)
    return render_template("thirdpage.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))
