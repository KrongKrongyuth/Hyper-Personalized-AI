import base64

py_dict = {
  "status": "success",
  "images": [
    
    "iVBORw0KGgoAAAANSUhE....."
  ]
}

# Access the first base64 string from the list
base64_image = py_dict["images"][0]

# Decode base64 string to bytes
image_bytes = base64.b64decode(base64_image)

# Save the image
with open("output.png", "wb") as f:
    f.write(image_bytes)
