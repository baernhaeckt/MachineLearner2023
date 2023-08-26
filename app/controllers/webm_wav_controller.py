import shutil
import subprocess
import os
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.post("/convert", tags=["webm wav conveter"], status_code=200, response_class=FileResponse)
def emotion_detection_audio(file: UploadFile):
    suffix = Path(file.filename).suffix

    with NamedTemporaryFile(delete=False, suffix=suffix) as webm:
        shutil.copyfileobj(file.file, webm)

        filename = f"/tmp/{uuid.uuid4()}.wav"
        command = ['ffmpeg', '-i', webm.name, filename]
        subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        if os.path.exists(filename):
            return FileResponse(filename)
        else:
            raise HTTPException(status_code=404, detail="File not found")

