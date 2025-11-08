from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.post("/caricature")
async def caricature(image: UploadFile = File(...), prompt: str = ""):
    # Dummy example: return the uploaded image unchanged
    contents = await image.read()
    return StreamingResponse(io.BytesIO(contents), media_type=image.content_type)
