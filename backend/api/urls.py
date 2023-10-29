from django.urls import path
from .views import simple_upload, task_check
urlpatterns = [
    path('upload_zip/', simple_upload),
    path('check_task/', task_check)
]
