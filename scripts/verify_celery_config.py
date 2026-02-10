import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.celery_app import app
    from src.core.config import get_settings
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this from the project root or scripts directory.")
    sys.exit(1)

def verify_celery():
    print("Verifying Celery Configuration...")
    try:
        settings = get_settings()
        print(f"Redis URL from settings: {settings.database.redis_url}")
    except Exception as e:
        print(f"Error loading settings: {e}")
        
    print(f"Celery Broker URL: {app.conf.broker_url}")
    print(f"Celery Result Backend: {app.conf.result_backend}")
    
    # Check Imports
    print(f"Celery Include: {app.conf.include}")
    
    # Ping Broker
    print("\nAttempting to connect to Broker...")
    try:
        with app.connection_for_write() as conn:
            conn.connect()
            print("✅ Successfully connected to Celery Broker!")
    except Exception as e:
        print(f"❌ Failed to connect to Broker: {e}")
        # Don't return, list tasks anyway

    # Check registered tasks
    print("\nRegistered Tasks:")
    found_tasks = []
    for task_name in sorted(app.tasks.keys()):
        if task_name.startswith('celery.'): continue
        print(f"- {task_name}")
        found_tasks.append(task_name)
        
    expected_tasks = ["src.celery_app.fan_out_forgetting", "src.celery_app.run_forgetting_task"]
    missing = [t for t in expected_tasks if t not in found_tasks]
    
    if missing:
        print(f"\n⚠️ Missing Expected Tasks: {missing}")
    else:
        print(f"\n✅ All expected tasks registered.")

if __name__ == "__main__":
    verify_celery()
