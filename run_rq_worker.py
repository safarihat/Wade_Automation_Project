import os
import sys
import django
from rq import Worker, Queue
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

# List of queues to listen on
# 'default' is the queue name used in django_rq.get_queue('default')
queues = [Queue('default', connection=redis_connection)]

print("--- Starting RQ Worker ---")
with Connection(redis_connection):
    worker = Worker(queues, connection=redis_connection)
    worker.work()
