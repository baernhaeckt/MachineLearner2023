import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, UploadFile

from services.emotion_detection_service import EmotionDetectionService

router = APIRouter()


@router.post("/audio", tags=["emotion detection"], status_code=200)
def emotion_detection_audio(file: UploadFile):
    suffix = Path(file.filename).suffix
    emotion_detection_service = EmotionDetectionService()

    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        detection_result = emotion_detection_service.detect_audio(tmp.name)

    return detection_result
