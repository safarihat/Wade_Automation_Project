import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
# It's good practice to do this at the top of the file.
from dotenv import load_dotenv
load_dotenv()


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# Load from environment variable, with a fallback for development.
# For production, ALWAYS set a unique, secret key in your environment.
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-your-secret-key')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = ['*'] # Added for development


# Application definition

INSTALLED_APPS = [
    # Django built-in apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',
    # Third-party apps
    'crispy_forms',
    'crispy_bootstrap5',
    'django_celery_results',
    # Local apps
    'wade_automation',
    'doc_generator',
    'invoice_reconciliation',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'wade_automation_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # This is the standard configuration. It tells Django to look for a
        # single 'templates' directory at the project's root level for any
        # project-wide templates. 'APP_DIRS': True will handle finding
        # templates inside individual app directories.
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'wade_automation_project.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    'default': {
        # Switched to the PostGIS backend for GeoDjango
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': 'wade_automation_db',
        'USER': 'wade_user',
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': 'localhost', # Or '127.0.0.1'
        'PORT': '5432',
    }
}

# --- Definitive GeoDjango Configuration for Windows/Conda ---
# This block ensures GeoDjango finds the correct GDAL/GEOS libraries within
# the Conda environment, bypassing potential version detection issues.
if os.name == 'nt':
    import sys
    conda_env_path = Path(sys.prefix)
    lib_dir = conda_env_path / 'Library' / 'bin'

    if lib_dir.is_dir():
        # Add the Conda library bin to the DLL search path. This is crucial
        # for GDAL to find its own dependencies (like PROJ, etc.).
        os.add_dll_directory(str(lib_dir))
        # Explicitly override the library paths Django will use.
        GDAL_LIBRARY_PATH = str(lib_dir / 'gdal.dll')
        GEOS_LIBRARY_PATH = str(lib_dir / 'geos_c.dll')

# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')


# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom user model


# Login redirect URL
LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/'

# LINZ API Key for Basemaps, loaded from .env file
LINZ_API_KEY = os.environ.get('LINZ_API_KEY')
if not LINZ_API_KEY:
    print("Warning: LINZ_API_KEY not found in .env file. Map features may not work.")

# Koordinates.com API Key for vector/raster queries, loaded from .env file
KOORDINATES_API_KEY = os.environ.get('KOORDINATES_API_KEY')
if not KOORDINATES_API_KEY:
    print("Warning: KOORDINATES_API_KEY not found in .env file. Geospatial analysis will fail.")

# LRIS (Manaaki Whenua) API Key for specific layers like Erosion Severity
LRIS_API_KEY = os.environ.get('LRIS_API_KEY')
if not LRIS_API_KEY:
    print("Warning: LRIS_API_KEY not found in .env file. Erosion Severity layer may not work.")

# Publicly available LINZ Basemaps API key for services like aerial imagery.
# This key does not require referrer restrictions.
# See: https://www.linz.govt.nz/data/linz-data-service/guides-and-documentation/linz-basemaps-service-api
LINZ_BASEMAPS_API_KEY = 'c01k45db6yn9zw23jjjy9kegjh9'

# Groq API Key for AI, loaded from .env file
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in .env file. Generative AI features may not work.")


# Email settings for contact form
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'webmaster@localhost')
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@example.com')


# --- Celery Configuration ---

# --- Production Best Practices ---
# 1. Message Broker: For production, use a robust, managed message broker like RabbitMQ or a managed Redis service.
# 2. Monitoring: Use a tool like Flower to monitor your Celery workers and tasks.
#    - To install: pip install flower
#    - To run: celery -A wade_automation_project flower --port=5555
# 3. Logging: Configure dedicated logging for Celery to a file for easier debugging.
#    - This can be configured in the LOGGING setting below.

# Use Redis for both development and production for reliability.
# The 'sqla+sqlite' broker is unreliable, especially with eventlet.
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

CELERY_TIMEZONE = TIME_ZONE # Use Django's timezone

# Store extended task metadata (e.g., arguments, start time, etc.)
CELERY_RESULT_EXTENDED = True 

# This setting silences a deprecation warning and ensures connection retries on startup.
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True

# --- Task Time Limits ---
# It's a best practice to set both soft and hard time limits to prevent tasks from hanging.
# A soft time limit raises an exception that the task can catch to clean up.
# A hard time limit forcefully terminates the task.
# NOTE: These are global defaults. They can be overridden on a per-task basis,
# which is often a better approach for tasks with different expected runtimes.
CELERY_TASK_TIME_LIMIT = 600  # Hard time limit of 10 minutes
CELERY_TASK_SOFT_TIME_LIMIT = 540 # Soft time limit of 9 minutes

# Disable ChromaDB telemetry
CHROMA_TELEMETRY_ENABLED = os.environ.get('CHROMA_TELEMETRY_ENABLED', 'False') == 'True'

# --- Crispy Forms Configuration ---
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"