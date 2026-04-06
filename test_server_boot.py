# test_server_boot.py
import os
import sys

# Ensure correct pathing
sys.path.append(os.getcwd())

print("--- TESTING SERVER BOOT ---")
try:
    from src.envs.satellite_env.server.app import app
    print("SUCCESS: FastAPI app initialized.")
except Exception as e:
    import traceback
    print("FAILED: App failed to initialize.")
    traceback.print_exc()
