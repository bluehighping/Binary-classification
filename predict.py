from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# ใส่ Channel Access Token และ Secret ที่ได้รับจาก LINE Developer Console
line_bot_api = LineBotApi('V6ACIs+J8ktXTVf8f9XS5v/fZx2HzA4rx/QYNFkD2DSP5kYqkUxQGq606eOs6+uZCTtgYpwZRCKPolCfgY7mZktu/U+7lvRl5s1D5UPB52A8IMyj+fmgm7WVIqpOx8ySdM1p2s507+v9bMXAt9GakQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('bdf99486c49ee382130aa9ab3bb557ab')

# โหลดโมเดล MobileNetV2
model = MobileNetV2(weights='imagenet')

# Route สำหรับ callback จาก LINE
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print(e)
        abort(400)

    return 'OK'

# Event handler สำหรับรับภาพจากผู้ใช้
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # รับภาพที่ผู้ใช้ส่งมา
    message_content = line_bot_api.get_message_content(event.message.id)
    img = Image.open(BytesIO(message_content.content))
    
    # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ทำนายผล
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # สร้างข้อความผลลัพธ์
    result = f"ผลการทำนาย:\n"
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        result += f"{i+1}. {label}: {score*100:.2f}%\n"

    # ส่งข้อความผลลัพธ์กลับไปยังผู้ใช้
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=result)
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
