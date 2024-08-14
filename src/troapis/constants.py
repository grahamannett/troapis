from os import environ

DEBUG_MODE = environ.get("DEBUG", "false").lower() in ["true", "1"]
SAVE_MODE = environ.get("SAVE", "true").lower() in ["true", "1"]
REQ_SAVE_DIR = environ.get("REQUEST_SAVE_DIR", "out/requests")

ENTRYPOINT_PATH = "model_entrypoint.py"
MODULE_NAME = "model_entrypoint"

device = environ.get("DEVICE", "cuda:0")
