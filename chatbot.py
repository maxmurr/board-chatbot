from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot import (
    LineBotApi, WebhookHandler
)
import os
from flask import Flask, request, abort
import warnings
from pythainlp import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
import random
from multiprocessing.sharedctypes import Value
from keras.models import load_model


warnings.filterwarnings('ignore')


app = Flask(__name__)

line_bot_api = LineBotApi(
    'FvqCc79VbcwRk6LHHEzRyYjGbeTU0eT2MgzoIkYDdOY6IUlnMaySVczF6GTgKuBRQUuNrfU9wOMDr5V9rzQ+0KnABieuDNqVlF5cLGFZlWkrV0eZbsHSRU+6PhXKe9idPoxYxK7LeNFL5lRCbWEPmQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('10dac3b4ec6908672de01ec31d5f7743')

# Webhook


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = word_tokenize(
        sentence, engine="deepcut", keep_whitespace=False)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    value = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['value'] == value:
            result = random.choice(i['responses'])
            break
    return result


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = event.message.text
    ints = predict_class(message)
    res = get_response(ints, intents)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)