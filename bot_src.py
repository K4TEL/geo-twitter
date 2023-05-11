#!/bin/python
import requests
import sqlite3
import os.path
from time import sleep
import json

from bert_predict import *

BOT_TOKEN = ""
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/"
PATH_TO_DB = "db.db"

START = "You can use this bot to access geolocation prediction model:\n" \
        "/predict _<text>_ (3-500 words) - get prediction results\n" \
        "/info - get information about the model"

INFO = "This model predicts the geolocation of short texts (less than 500 words) in a form of " \
       "two-dimensional distributions also referenced as the Gaussian Mixture Model (GMM).\n" \
       "\nBERT Regression model for 5 outcome(s):\nCoordinates: 10\nWeights: 5\nCovariances: 5 (matrix type: spher)\n" \
       "\n\U0001F464 [GitHub project repo](https://github.com/K4TEL/geo-twitter.git)\n" \
       "\n\U0001F917 [HuggingFace model repo](https://huggingface.co/k4tel/geo-bert-multilingual)\n" \
       "\n\U0001F4F0 [arXiv paper preprint](https://arxiv.org/pdf/2303.07865.pdf)"


# parsing functions
# user message parsing
def parse_cmd(msg):
    text = msg["message"]["text"]
    if len(text) == 0:  # empty text
        return None, None
    if text[0] != "/":  # not a command
        return None, text
    text = text[1:]  # crop "/"
    if " " in text:  # command with an argument
        split = text.split(" ")
        return split[0], " ".join(split[1:])
    else:  # command without an argument
        return text, None


# user profile parsing
def parse_user(msg):
    user = msg["message"]["from"]
    return user["username"], f'{user["first_name"]} {user["last_name"]}'


# bot updates parsing
def parse_updates(update_id=0):  # -> new update_id, List[postDict]
    result = requests.post(f'{BASE_URL}getUpdates?update_id={update_id}').json()
    if result["ok"]:
        msgs = [item for item in result["result"] if item["update_id"] > update_id]  # get unprocessed
        if len(msgs) != 0:
            update_id = max(map(lambda msg: msg.get('update_id', 0), msgs))  # update last update_id
        return msgs, update_id
    return None


# DB functions:
# load DB connection
def load_DB(db_path):  # -> DB Connection
    is_created = os.path.exists(db_path)
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(e)
        return None
    # create db if needed
    # columns: update_id, username, user tag, command, argument/text, bot response)
    if not is_created:
        conn.execute('''CREATE TABLE LOGS
                 (ID      INT  PRIMARY KEY NOT NULL,
                  USER    TEXT             NOT NULL,
                  TAG     TEXT             NOT NULL,
                  CMD     TEXT             NULL,
                  TEXT    TEXT             NULL,
                  OUT     TEXT             NULL);''')
    return conn


# log request and response to DB
def log_to_DB(conn, msg, response):
    usertag, username = parse_user(msg)
    command, arg = parse_cmd(msg)
    update_id = msg["update_id"]

    if conn is not None:
        cur = conn.cursor()
        try:
            sql_umid = ''' INSERT INTO LOGS(ID, USER, TAG, CMD, TEXT, OUT)
              VALUES(?,?,?,?,?,?) '''
            cur.execute(sql_umid, (update_id, usertag, username, command, arg, response))
            conn.commit()
            print(f"Message {update_id} from {usertag} was logged in DB")
        except Exception as e:
            print("Some kind of error occurred")
            print(e)


class BotChat():
    def __init__(self, chat_id, db_conn, model):
        self.chat_id = chat_id
        self.db_conn = db_conn
        self.model = model

        self.text_base_url = f"{BASE_URL}sendMessage?chat_id={self.chat_id}&parse_mode=Markdown"
        self.image_base_url = f"{BASE_URL}sendPhoto?chat_id={self.chat_id}&parse_mode=Markdown"

        self.handlers = {
            "start": self.handle_start,
            "info": self.handle_info,
            "predict": self.handle_predict,
        }

    # send text to chat
    def send_text(self, text):
        result = requests.get(f"{self.text_base_url}&text={text}").json()
        return result["result"] if result["ok"] else None

    # send image file with text caption to chat
    def send_image(self, filename, caption=""):
        result = requests.post(f"{self.image_base_url}&caption={caption}",
                               files={"photo": open(filename, "rb")}).json()
        return result["result"] if result["ok"] else None

    # routing user request to the command handlers
    def process_request(self, msg):
        command, arg = parse_cmd(msg)
        print(self.chat_id, command, arg)
        if command in self.handlers:
            self.handlers[command](arg, msg)
        else:
            self.send_text("Unknown command!")

    # /info handler
    def handle_info(self, arg, msg):
        if self.send_text(INFO):
            log_to_DB(self.db_conn, msg, "info")

    # /start handler
    def handle_start(self, arg, msg):
        if self.send_text(START):
            log_to_DB(self.db_conn, msg, "start")

    # /predict <text> handler
    def handle_predict(self, arg, msg):
        if not arg or len(arg) == 0:
            response = "Text argument is empty!"
            if self.send_text(response):
                log_to_DB(self.db_conn, msg, response)

        elif len(self.model.tokenizer.tokenize(arg)) > 512:
            response = "Text length exceeds 512 tokens!"
            if self.send_text(response):
                log_to_DB(self.db_conn, msg, response)

        else:
            self.send_text("Estimating geolocation and generating GMM plot...")
            result = text_prediction(self.model, arg)
            response = result.result_to_text()
            filename = f"{msg['update_id']}.png"

            visual = ResultVisuals(result)
            visual.text_map_result(filename)

            if self.send_image(filename, response):
                log_to_DB(self.db_conn, msg, response)
                os.remove(filename)


# run the bot
def bot_loop():
    model = load_model()
    conn_DB = load_DB(PATH_TO_DB)
    chats = {}

    if conn_DB is not None:
        cur = conn_DB.cursor()
        a = cur.execute("""SELECT MAX(ID) FROM LOGS;""")
        row = cur.fetchone()

    update_id = int(row[0]) if row and row[0] else 779342409

    print("Listening...")
    while conn_DB is not None:
        try:
            msgs, update_id = parse_updates(update_id)
            if msgs:
                print(msgs)
                for msg in msgs:
                    chat_id = msg["message"]["chat"]["id"]
                    if chat_id not in chats:
                        chats[chat_id] = BotChat(chat_id, conn_DB, model)

                    chat = chats[chat_id]
                    chat.process_request(msg)
        except Exception as e:
            print(e)
        finally:
            sleep(1)


if __name__ == "__main__":
    bot_loop()
