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
from linebot import LineBotApi
from linebot.models import TextSendMessage


warnings.filterwarnings('ignore')


app = Flask(__name__)

line_bot_api = LineBotApi(
    'FvqCc79VbcwRk6LHHEzRyYjGbeTU0eT2MgzoIkYDdOY6IUlnMaySVczF6GTgKuBRQUuNrfU9wOMDr5V9rzQ+0KnABieuDNqVlF5cLGFZlWkrV0eZbsHSRU+6PhXKe9idPoxYxK7LeNFL5lRCbWEPmQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('10dac3b4ec6908672de01ec31d5f7743')
userId = "U0c537ff4217faf7fcf64b6c4b78b212f"


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


# clean up sentence from user
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(
        sentence, engine="deepcut", keep_whitespace=False)
    # sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# create bag of words
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


# def get_response2(intents_list, intents_json):
#     value = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['value'] == value:
#             result2 = i['responses2'][0]
#             break
#     return result2


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = event.message.text
    ints = predict_class(message)
    res = get_response(ints, intents)
    # res2 = get_response2(ints, intents)
    if res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย สำนักทรัพย์สิน เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา"or res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย สำนักบริการคอมพิวเตอร์ เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา หรือจะติดต่อสำนักบริการคอมพิวเตอร์ โดยตรงได้ที่ FACEBOOK | www.ocs.ku.ac.th | LINE@: @GQV5600M | HELPDESK 025620952-6 ต่อ 622541-43" or res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย สำนักกีฬา เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา" or res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย สำนักการศึกษา เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา หรือจะติดต่อ สำนักการศึกษา โดยตรงได้ที่ https://www.facebook.com/kuregistrar ในการติดต่อ สบศ. โปรดแจ้งรหัสนิสิต ชื่อ-นามสกุล และอีเมล KU-Google ทุกครั้งที่ติดต่อด้วยนะคะ" or res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย กองกิจการนิสิต เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา หรือจะติดต่อ กองกิจการนิสิต และติดตามข้อมูลข่าวสารโดยตรงได้ที่  https://www.facebook.com/SAKUkasetsart" or res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย กองยานพาหนะ อาคารและ สถานที่ เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา หรือจะติดต่อ กองยานพาหนะ อาคารและ สถานที่ โดยตรงได้ที่ vehicle.ku.ac.th" or res == " ปัญหานี้อยู่ในความรับผิดชอบของฝ่าย Happy Place Center เราจะดำเนินการส่งข้อร้องเรียนไปยังหน่วยงานที่เกี่ยวข้อง ขอบคุณที่แจ้งปัญหาเข้ามา หรือจะติดต่อ Happy Place Center โดยตรงได้ที่ https://www.facebook.com/KUHappyPlaceCenter":
        line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="ปัญหา " + message + res))
    else:
        line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res))

    # if res2 != None:
    #     try:
    #         line_bot_api.push_message(
    #             userId, TextSendMessage(text=res2))
    #     except:
    #         print("Error")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
