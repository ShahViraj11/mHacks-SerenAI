from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://backupofamrit:GrJDmcTLkqxnR7Bo@aanlysiscluster.vwrt8og.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
def create():
    db = client['user_info']
    coll = db["addressconsole"]
    item = {
        "_id": "18181",
        "announcements": [],
        "attending": 0,
        "space": 999,
        "waitlist": 0,
    }
    coll.insert_one(item)
def search():
    db = client['user_info']
    coll = db["addressconsole"]
    review_dict = coll.find_one({"_id": "69696969"})
    print(review_dict["attending"])
create()
search()