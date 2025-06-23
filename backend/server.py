import requests
import base64
from PIL import Image
import io

url = "https://41f3-34-124-238-99.ngrok-free.app/segmentar/" 

def get_image_inference(body):

    response = requests.post(url, files=body)
    
    if  response.status_code == 200:
            data = response.json()
            img_bytes = base64.b64decode(data["image_b64"])
            img = Image.open(io.BytesIO(img_bytes))
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
    
    return data