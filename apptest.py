from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


app = Flask(__name__)
uri = "mongodb+srv://backupofamrit:GrJDmcTLkqxnR7Bo@aanlysiscluster.vwrt8og.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))


def search():
    db = client['sample_airbnb']
    coll = db["listingsAndReviews"]
    review_dict = coll.find_one({"_id": "69696969"})
    return review_dict["attending"]

@app.route('/')
def landing():
    data = search()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
