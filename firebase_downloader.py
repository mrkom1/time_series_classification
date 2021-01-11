from typing import List

from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
import json


class FirebaseClient:

    def __init__(self, private_key_file: str, config_file: str):
        # initialize sdk
        cred = credentials.Certificate(private_key_file)
        firebase_admin.initialize_app(cred)
        self._firestore_db = firestore.client()
        with open("firebase_creds/configs.json", "r") as read_file:
            self._config_firebase = json.load(read_file)

    @property
    def firestore_db(self):
        return self._firestore_db

    @property
    def config_firebase(self):
        return self._config_firebase

    def get_sess_ids_by_file_id(self,
                                file_id: str,
                                ignored_users: List[str] = []) -> dict:
        """Find all sessions id`s for needed file id

        Parameters
        ----------
        file_id : str
            file id
        ignored_users : List[str], optional
            users id`s, whos sessions need to be ignored, by default []

        Returns
        -------
        dict
            sess id`s with some data about camera and screen
        """
        sess_ids = {}
        sessions = list(self.firestore_db.collection(u'sessions').get())
        for sess in sessions:
            sess_meta = sess.to_dict()
            if sess_meta['userId'] in ignored_users:
                continue
            if sess_meta['contentId'] == file_id:
                sess_ids[sess.id] = {
                    "camera": sess_meta["tracker"]["setup"]["camera"],
                    "screen": sess_meta["tracker"]["setup"]["screen"]
                }
        return sess_ids

    def delete_sess(self, sess_id: str, delete_data: bool = True):
        """Delete sess record

        Parameters
        ----------
        sess_id : str
            session id in db
        delete_data : bool, optional
            delete sess files from storage, by default True
        """
        self.firestore_db.collection(u'sessions').document(sess_id).delete()
        print(f"{sess_id} removed from db")

        if delete_data:
            firebase = pyrebase.initialize_app(self.config_firebase)
            storage = firebase.storage()
            all_files = storage.list_files()
            for file in all_files:
                if file.name.startswith("sessions") and \
                    (file.name.endswith("json") or
                    file.name.endswith("csv")) and \
                    (file.name.split("/")[1] == sess_id):
                    storage.delete(file.name)
                    print(f"🗑 {file.name} deleted")

    def delete_all_sess_for_user(self, 
                                 userid: str, 
                                 delete_data: bool = True):
        """
        Delete all sessions for user by his id
        """
        sessions = list(self.firestore_db.collection(u'sessions').get())
        for sess in sessions:
            sess_meta = sess.to_dict()
            if sess_meta['userId'] == userid:
                self.delete_sess(sess.id, delete_data)

    def delete_all_sess(self, 
                        delete_data: bool = True):
        """
        Delete all sessions
        """
        sessions = list(self.firestore_db.collection(u'sessions').get())
        for sess in sessions:
            self.delete_sess(sess.id, delete_data)

    def get_sess_files(self) -> list:
        """
        Return blobs of sessions from Cloud Firestore
        """
        firebase = pyrebase.initialize_app(self.config_firebase)
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

    def download_sess_by_ids(self,
                             sessions: dict,
                             dowload_dir: str = "cloud"):
        """
        Download sessions by its id`s on your local drive
        """
        sess_files = self.get_sess_files()
        root_path = Path(dowload_dir)
        for sess_id, sess_meta in sessions.items():
            sess_path = root_path / str(sess_id)
            sess_path.mkdir(parents=True, exist_ok=True)
            print(f"-> {sess_path}")
            # write meta file
            with open(sess_path/'meta.json', 'w') as fp:
                json.dump(sess_meta, fp)
            # download sess`s
            for sess in sess_files:
                if sess_id in sess.name:
                    try:
                        filename = sess.name.split("/")[-1]
                        sess.download_to_filename(sess_path / filename)
                        print(f"✅ {filename} download Complete")
                    except Exception:
                        print(f'❌ {filename} download Failed')


if __name__ == '__main__':
    # generate key here https://bit.ly/3h8uYD7
    private_key_file = "firebase_creds/beehiveor-eyepass-firebase-adminsdk-u47mu-60369f4307.json"
    config_file = "firebase_creds/configs.json"
    firebase_cli = FirebaseClient(private_key_file, config_file)

    firebase_cli.delete_all_sess()
    # # download firebase sessions
    # print("images")
    # sess_ids = firebase_cli.get_sess_ids_by_file_id("5IFyKZXhD4uPN8Cd9QpK")
    # firebase_cli.download_sess_by_ids(sess_ids, "data/non-reading")
    # print("text")
    # sess_ids = firebase_cli.get_sess_ids_by_file_id("LJwXzrOlk9bzgWDFzpSV")
    # firebase_cli.download_sess_by_ids(sess_ids, "data/reading")
    # print("video")
    # sess_ids = firebase_cli.get_sess_ids_by_file_id("igBlC9lDwNPWd5yCMkyq")
    # firebase_cli.download_sess_by_ids(sess_ids, "data/non-reading")
    # print("video(cat)")
    # sess_ids = firebase_cli.get_sess_ids_by_file_id("heAGRyb6V82xmPjuiDFt")
    # firebase_cli.download_sess_by_ids(sess_ids, "data/non-reading")

