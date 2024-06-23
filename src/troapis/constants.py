from os import environ

DEBUG_MODE = environ.get("DEBUG", "false").lower() in ["true", "1"]
ENTRYPOINT_PATH = "model_entrypoint.py"
MODULE_NAME = "model_entrypoint"
