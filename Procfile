web: gunicorn -c gunicorn_config.py main:app
worker: celery -A celery_worker worker --loglevel=info 