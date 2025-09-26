import os
import logging
from celery import Celery
from celery.signals import task_prerun, task_postrun, worker_init, worker_shutdown
from django.db import close_old_connections

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wade_automation_project.settings')

app = Celery('wade_automation_project')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# --- Logging ---
logger = logging.getLogger(__name__)


# --- Graceful Worker Shutdown ---
@worker_init.connect
def on_worker_init(**kwargs):
    """Log worker startup."""
    logger.info("Celery worker started.")

@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Log worker shutdown and perform any cleanup."""
    logger.info("Celery worker shutting down gracefully.")
    # Add any cleanup tasks here, like closing external connections.


# --- Database Connection Handling for Concurrency ---
# This is crucial for preventing "DatabaseWrapper" errors when using eventlet or gevent.
@task_prerun.connect
def on_task_prerun(*args, **kwargs):
    """Close old database connections before the task runs."""
    close_old_connections()

@task_postrun.connect
def on_task_postrun(*args, **kwargs):
    """Close old database connections after the task runs."""
    close_old_connections()
