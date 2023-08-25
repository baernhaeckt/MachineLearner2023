from fastapi import APIRouter

import config

router = APIRouter()


@router.get("/", tags=["api health"], status_code=200)
def touch():
    return f"Model API is running on ENV {config.settings.ENV}"
