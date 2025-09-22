# This is the crucial fix for eventlet concurrency.
# It must be called before any other modules (like socket or requests) are imported.
import eventlet
eventlet.monkey_patch()

import os
from wade_automation_project.celery import app

def main():
    """
    This script ensures monkey-patching is done before starting the Celery worker.
    It programmatically runs the equivalent of:
    `celery -A wade_automation_project worker -l info -P eventlet`
    """
    # Set the default Django settings module for the 'celery' program.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wade_automation_project.settings')
    app.worker_main(['worker', '--loglevel=info', '--pool=eventlet'])

if __name__ == '__main__':
    main()