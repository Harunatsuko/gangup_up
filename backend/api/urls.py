from django.urls import path
from .views import simple_upload
urlpatterns = [
    path('upload_zip/', simple_upload)
]
