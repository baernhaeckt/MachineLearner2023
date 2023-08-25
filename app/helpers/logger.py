import logging


# convenience ai_management_helpers to be used in the other modules/files
# 'main' is to be replaced with the name of the app which is, by default, the name of the main python file (main.py, app.py, application.py etc)

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("main")

logger.info("Logging initialised")
