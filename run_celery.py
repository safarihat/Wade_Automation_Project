# This is the crucial fix for gevent concurrency.
# It must be called before any other modules (like socket or requests) are imported.
import gevent.monkey
gevent.monkey.patch_all()

import os
from wade_automation_project.celery import app

def main():
    """
    This script ensures monkey-patching is done before starting the Celery worker.
    It programmatically runs the equivalent of:
    `celery -A wade_automation_project worker -l info -P gevent -c 4 --prefetch-multiplier=1`
    """
    # Set the default Django settings module for the 'celery' program.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wade_automation_project.settings')
    # Use gevent for concurrency, with a concurrency of 4 and prefetch multiplier of 1
    app.worker_main(['worker', '--loglevel=info', '--pool=gevent', '-c', '4', '--prefetch-multiplier=1'])

if __name__ == '__main__':
    main()