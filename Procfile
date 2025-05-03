web: gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
worker: celery -A celery_worker worker --loglevel=info 