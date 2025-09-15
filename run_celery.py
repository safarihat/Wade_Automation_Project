import os
import sys
from celery.bin import celery

if __name__ == '__main__':
    # Set the default Django settings module for the 'celery' program.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wade_automation_project.settings')

    # We must explicitly tell Celery which app instance to use.
    # The app is defined in 'wade_automation_project/celery.py'.
    # The standard way to do this is with the '-A' or '--app' command-line flag.
    # We will prepend this to the arguments passed to the script.
    if '-A' not in sys.argv and '--app' not in sys.argv:
        sys.argv.insert(1, '-A')
        sys.argv.insert(2, 'wade_automation_project')

    # Execute the Celery command-line utility.
    celery.main()