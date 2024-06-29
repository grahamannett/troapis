from os import environ

DEBUG_MODE = environ.get("DEBUG", "false").lower() in ["true", "1"]

ENTRYPOINT_PATH = "model_entrypoint.py"
MODULE_NAME = "model_entrypoint"

SAVE_MODE = environ.get("DEBUG", "true").lower() in ["true", "1"]
REQ_SAVE_DIR = environ.get("REQUEST_SAVE_DIR", "out/requests")

# constant but i dont like it uppercase
device = environ.get("DEVICE", "cuda:0")
