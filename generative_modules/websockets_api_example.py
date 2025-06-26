# websockets_api_example.py

import websocket
import uuid
import json
import urllib.request
import urllib.parse
import requests
import random
from PIL import Image
import io
from typing import List, Union

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            continue  # skip binary previews

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def upload_file(file_data: Union[bytes, io.BytesIO], subfolder="", overwrite=False) -> str:
    try:
        body = {"image": ('uploaded_image.png', file_data)}
        data = {}
        if overwrite:
            data["overwrite"] = "true"
        if subfolder:
            data["subfolder"] = subfolder

        resp = requests.post(f"http://{server_address}/upload/image", files=body, data=data)
        if resp.status_code == 200:
            data = resp.json()
            path = data["name"]
            if "subfolder" in data and data["subfolder"]:
                path = f"{data['subfolder']}/{path}"
            return path
        else:
            raise Exception(f"Upload failed: {resp.status_code} - {resp.reason}")
    except Exception as error:
        raise RuntimeError(f"Upload error: {error}")

def generate_image_from_prompt_and_file(
    file_obj: Union[bytes, io.BytesIO],
    workflow_path: str,
    positive_prompt: str,
    negative_prompt: str,
    ci_positive_prompt: str = "",
    ci_negative_prompt: str = "",
    upload_subfolder: str = "",
    overwrite: bool = True
) -> List[bytes]:
    # Upload image
    comfyui_path_image = upload_file(file_obj, subfolder=upload_subfolder, overwrite=overwrite)

    # Load workflow from file
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_data = f.read()
    workflow = json.loads(workflow_data)

    # Set prompts
    workflow["6"]["inputs"]["text"] = positive_prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["24"]["inputs"]["text"] = ci_positive_prompt
    workflow["25"]["inputs"]["text"] = ci_negative_prompt

    # Random seed
    seed = random.randint(1, 1000000000)
    workflow["3"]["inputs"]["seed"] = seed

    # Set uploaded image path in the workflow
    workflow["10"]["inputs"]["image"] = comfyui_path_image

    # Connect WebSocket and retrieve images
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    images = get_images(ws, workflow)

    result_images = []
    for node_id in images:
        for image_data in images[node_id]:
            result_images.append(image_data)

    return result_images