# from dotenv import load_dotenv
import os

print(os.environ["HF_TOKEN"])
env_token = os.environ.get("HF_TOKEN")
print(env_token)