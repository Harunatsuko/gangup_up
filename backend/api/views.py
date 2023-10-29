from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from celery import current_app
import uuid
import os
from .tasks import input_archive
from .models import ArchiveModel, TaskModel


@csrf_exempt
def simple_upload(request):
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file_model = ArchiveModel(file=str(filename))
        file_model.save()

        task_id = input_archive.delay(
            os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, filename),
            os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, f'unzipped/{str(uuid.uuid4())}'),
            file_model.pk
        )
        task_model = TaskModel(task_id=task_id, input_file=file_model, task_type='unzip')
        task_model.save()
        return JsonResponse({'status': 'ok', 'url': uploaded_file_url, 'task_id': str(task_id)})


@csrf_exempt
def task_check(request):
    if request.method == 'POST':
        task_id = request.POST.get('id')
        task = current_app.AsyncResult(task_id)
        response_data = {'task_status': task.status, 'task_id': task.id}
        return JsonResponse(response_data)
