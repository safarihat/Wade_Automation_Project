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
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis', # Add for GeoDjango functionality
    'wade_automation',
    'doc_generator',
    'invoice_reconciliation',
    'django_celery_results',
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
        'DIRS': [BASE_DIR / 'wade_automation' / 'templates' / 'wade_automation'],
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
        # Switched to the SpatiaLite backend for GeoDjango
        'ENGINE': 'django.contrib.gis.db.backends.spatialite',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# --- GeoDjango Configuration ---
# On Windows, you often need to explicitly point to the GIS libraries.
# This configuration is for a Conda environment.
if os.name == 'nt':
    import sys
    # sys.prefix points to the root of the current Python environment.
    # In a Conda env, the required DLLs are in 'Library/bin'.
    conda_env_path = Path(sys.prefix)
    lib_dir = conda_env_path / 'Library' / 'bin'

    if lib_dir.is_dir():
        # Add the library directory to the DLL search path. This is the modern
        # and recommended way to handle DLL dependencies on Windows.
        os.add_dll_directory(str(lib_dir))

        # Explicitly set the paths for the main libraries. This is the most
        # reliable way to ensure GeoDjango finds them, as it looks for these
        # variables directly.
        GDAL_LIBRARY_PATH = str(lib_dir / 'gdal.dll')
        GEOS_LIBRARY_PATH = str(lib_dir / 'geos_c.dll')
        SPATIALITE_LIBRARY_PATH = str(lib_dir / 'mod_spatialite.dll')

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

# Groq API Key for AI, loaded from .env file
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in .env file. Generative AI features may not work.")


# --- Celery Configuration ---
# Use Redis for both development and production for reliability.
# The 'sqla+sqlite' broker is unreliable, especially with eventlet.
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

CELERY_TIMEZONE = TIME_ZONE # Use Django's timezone
CELERY_RESULT_EXTENDED = True # To store more metadata about tasks
# This setting silences a deprecation warning and ensures connection retries on startup.
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
