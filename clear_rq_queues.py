import os
import sys
from rq import Queue
from redis import Redis

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Provide a connection to Redis
redis_connection = Redis(host='localhost', port=6379, db=0)

# List of queues to clear
queue_names = ['high', 'default', 'failed']

print("--- Clearing RQ Queues ---")

for queue_name in queue_names:
    q = Queue(queue_name, connection=redis_connection)
    num_jobs = q.count
    if num_jobs > 0:
        print(f"Emptying queue '{queue_name}' which has {num_jobs} jobs.")
        q.empty()
        print(f"Queue '{queue_name}' is now empty.")
    else:
        print(f"Queue '{queue_name}' is already empty.")

print("--- Finished Clearing RQ Queues ---")
