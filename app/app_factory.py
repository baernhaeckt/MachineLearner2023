from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from controllers import health_controller
from helpers.logger import logger


def app_factory() -> FastAPI:
    # Init fast api
    app: FastAPI = FastAPI(title="Emotion detection")

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # Add router
    app.include_router(health_controller.router, prefix="/api/v1/health")

    # Logging
    logger.info(f"Starting app with profile: {config.settings.ENV}")

    return app
