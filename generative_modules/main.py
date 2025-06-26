from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import base64
from websockets_api_example import generate_image_from_prompt_and_file
import io
from typing import List, Union

app = FastAPI()

@app.post("/generate-image")
def generate_image(
    image: str,
    positive_prompt: str,
    negative_prompt: str,
    ci_positive_prompt: str,
    ci_negative_prompt: str
):
    try:
        print(image)
        # Generate images (list of bytes)
        generated_images = generate_image_from_prompt_and_file(
            file_obj=image,
            workflow_path="MMS_hyper.json",
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            ci_positive_prompt=ci_positive_prompt,
            ci_negative_prompt=ci_negative_prompt
        )

        # Encode images to base64
        base64_images = [base64.b64encode(img).decode("utf-8") for img in generated_images]

        return JSONResponse(
            content={
                "status": "success",
                "images": base64_images
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})