import os
import sys

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "video_summarization",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
)

max_tasks_per_child = os.environ.get("CELERY_MAX_TASKS_PER_CHILD")
if max_tasks_per_child:
    celery_app.conf.worker_max_tasks_per_child = int(max_tasks_per_child)

# On macOS, prefork + heavy ML/video libs can crash workers with SIGABRT.
# Default to solo unless the user explicitly overrides via env vars.
if sys.platform == "darwin":
    celery_app.conf.worker_pool = os.environ.get("CELERY_WORKER_POOL", "solo")
    celery_app.conf.worker_concurrency = int(
        os.environ.get("CELERY_WORKER_CONCURRENCY", "1")
    )
