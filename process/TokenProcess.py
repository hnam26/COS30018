import os
import wandb
from huggingface_hub import login
from dotenv import load_dotenv
class Token:
    def __init__(self):
        load_dotenv()
        wandb.login(key = os.getenv("WANDB_API_KEY"))
        login(token = os.getenv("HUGGINGFACE_API_KEY"))