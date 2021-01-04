from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
import json
from pprint import pprint

# initialize sdk
# generate key here https://bit.ly/3h8uYD7
private_key_file = "firebase_creds/beehiveor-eyepass-firebase-adminsdk-u47mu-60369f4307.json"
cred = credentials.Certificate(private_key_file)
firebase_admin.initialize_app(cred)
with open("firebase_creds/configs.json", "r") as read_file:
    config = json.load(read_file)

def get_sess_ids_by_file_id(file_id: str) -> list:
    # initialize firestore instance
    firestore_db = firestore.client()

    sess_ids = {}
    sessions = list(firestore_db.collection(u'sessions').get())
    for sess in sessions:
        sess_meta = sess.to_dict()
        # print(sess_meta)
        if sess_meta['contentId'] == file_id:
            sess_ids[sess.id] = {
                "camera": sess_meta["tracker"]["setup"]["camera"],
                "screen": sess_meta["tracker"]["setup"]["screen"]
            }
    return sess_ids


def get_sess_files() -> dict:
    """
    Return blobs of sessions from Cloud Firestore
    """
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    # sess_files = storage.child("sessions").list_files()  # this child sh#t doesn`t work (>_<)
    all_files = storage.list_files()
    sess_files = []
    for file in all_files:
        if file.name.startswith("sessions") and \
           (file.name.endswith("json") or
            file.name.endswith("csv")):
            sess_files.append(file)
    return sess_files


def download_sess_by_ids(sessions: dict,
                         subdir: str = ""):
    """
    Download sessions by its id`s on your local drive
    `cloud/{subdir}/sessions`
    """
    sess_files = get_sess_files()
    root_path = Path("cloud")
    if subdir:
        root_path = root_path / subdir
    for sess_id, sess_meta in sessions.items():
        sess_path = root_path / (f"sessions/{sess_id}/")
        sess_path.mkdir(parents=True, exist_ok=True)
        print(f"-> {sess_path}")
        with open(sess_path/'meta.json', 'w') as fp:
            json.dump(sess_meta, fp)
        for sess in sess_files:
            if sess_id in sess.name:
                try:
                    sess.download_to_filename(root_path / sess.name)
                    print(f"✅ {sess.name} download Complete")
                except Exception:
                    print(f'❌ {sess_id} download Failed')

if __name__ == '__main__':
    print("images")
    sess_ids = get_sess_ids_by_file_id("5IFyKZXhD4uPN8Cd9QpK")
    download_sess_by_ids(sess_ids, "non-reading")
    print("text")
    sess_ids = get_sess_ids_by_file_id("LJwXzrOlk9bzgWDFzpSV")
    download_sess_by_ids(sess_ids, "reading")
    print("video")
    sess_ids = get_sess_ids_by_file_id("igBlC9lDwNPWd5yCMkyq")
    download_sess_by_ids(sess_ids, "non-reading")
    print("video(cat)")
    sess_ids = get_sess_ids_by_file_id("heAGRyb6V82xmPjuiDFt")
    download_sess_by_ids(sess_ids, "non-reading")
