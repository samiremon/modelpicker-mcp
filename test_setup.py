"""
Step 1 Verification: Test Environment and Hugging Face Connection.
"""
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Fix Windows console encoding for UTF-8 characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

print("=" * 50)
print("ModelPicker MCP - Environment Verification")
print("=" * 50)

if not hf_token or hf_token == "your_huggingface_token_here":
    print("[!] HF_TOKEN not found in .env file.")
    print("👉 Please create a .env file with: HF_TOKEN=hf_your_actual_token")
    print("👉 Get a free Read token from: https://huggingface.co/settings/tokens")
else:
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname")
        print("[+] Successfully connected to Hugging Face Cloud!")
        print(f"[+] Authenticated as: {username}")
        print("[+] Environment is 100% ready for Step 2!")
    except Exception as e:
        print(f"[-] Failed to authenticate with Hugging Face: {e}")
print("=" * 50)
