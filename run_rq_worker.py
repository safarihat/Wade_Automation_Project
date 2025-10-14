import os
import sys
import django
from rq import SimpleWorker, Queue
from redis import Connection
from redis import Redis

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wade_automation_project.settings')
django.setup()

# Provide a connection to Redis
# Ensure this matches your settings.py RQ_QUEUES configuration
redis_connection = Redis(host='localhost', port=6379, db=0)

# List of queues to listen on. Order can matter for priority.
queues = [Queue('high', connection=redis_connection), Queue('default', connection=redis_connection)]

print("--- Starting RQ Worker ---")
# Use SimpleWorker for Windows compatibility (no os.fork)
worker = SimpleWorker(queues, connection=redis_connection)
worker.work()
