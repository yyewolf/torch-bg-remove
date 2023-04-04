from app.background import remove_background
import numpy as np
from fastapi import FastAPI, File, Response
from PIL import Image
import io
import cv2

app = FastAPI()

def background_remove(img):
    test_image = Image.open(io.BytesIO(img)).convert('RGB')
    test_image = np.array(test_image)
    test_image = remove_background(test_image)
 
    # aimed width is md.IMG_WIDTH, aimed height is md.IMG_HEIGHT
    if test_image.shape[0] > test_image.shape[1]:
        # height is greater than width
        pad_size = (test_image.shape[0] - test_image.shape[1]) // 2
        padded_image = cv2.copyMakeBorder(test_image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # width is greater than height
        pad_size = (test_image.shape[1] - test_image.shape[0]) // 2
        padded_image = cv2.copyMakeBorder(test_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    resized_image = cv2.resize(padded_image, (224, 224))
        
    # convert back into bytes to return it
    img = Image.fromarray(resized_image)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/transforms/background_removal", responses={
    200: {
        "content": {"image/png": {}}
    }
}, response_class=Response)
async def predict(upload: bytes = File(...)):
    # Get the image from the request
    if not upload:
        return {"message": "No upload file sent"}
    return Response(content=background_remove(upload), media_type="image/png")
