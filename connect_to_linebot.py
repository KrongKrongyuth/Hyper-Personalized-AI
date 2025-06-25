from analytics_modules.tools import ClusteringCustomer

from linebot.v3 import WebhookHandler, WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import (
    ApiClient, 
    MessagingApi,
    MessagingApiBlob,
    Configuration, 
    ReplyMessageRequest, 
    TextMessage
)

from fastapi import FastAPI, Request, HTTPException, Header
from dotenv import load_dotenv; load_dotenv()

import os

api = FastAPI()
agent = ClusteringCustomer()

get_access_token = os.getenv('CHANNEL_ACCESS_TOKEN')
get_channel_secret = os.getenv('CHANNEL_SECRET')

parser = WebhookParser(channel_secret=get_channel_secret)
configuration = Configuration(access_token=get_access_token)
handler = WebhookHandler(channel_secret=get_channel_secret)

@api.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    body_str = body.decode('utf-8')
    try:
        handler.handle(body_str, x_line_signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature.")

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_query = event.message.text
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        reponse_message = agent.apply_KMeans(objective=user_query)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reponse_message)]
            )
        )

@handler.add(MessageEvent, message=ImageMessageContent)
def handel_image(event):
    message_id = event.message.id
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        messaging_api_blob = MessagingApiBlob(api_client)
        image_data = messaging_api_blob.get_message_content(message_id)
        
        file_path = f"./line_image/received_image_{message_id}.jpg"
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="ได้รับรูปภาพแล้วครับ✅")]
            )
        )