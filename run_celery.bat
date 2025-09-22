@echo off
:: This script starts the Celery worker by calling a Python script that
:: correctly applies the eventlet monkey-patch before anything else.
::
:: Usage:
::   run_celery.bat

echo "--- Starting Celery Worker with Eventlet Pool ---"
python run_celery.py