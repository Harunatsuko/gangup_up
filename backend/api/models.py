from django.db import models
from django.conf import settings
from django.utils import timezone
import os

class ArchiveModel(models.Model):
    """
    Модель для входных файлов
    """
    file = models.FileField()
    unzipped_path = models.CharField(default=os.path.join(settings.BASE_DIR, 'media/empty/'), max_length=1024)
    show_rgb = models.BooleanField(default=True)
    show_tiff = models.BooleanField(default=False)

    def delete(self, *args, **kwargs):
        self.file.delete()
        super(ArchiveModel, self).delete(*args, **kwargs)


class TaskModel(models.Model):
    """
    Модель для тасок Celery
    """
    timestamp = models.DateTimeField(default=timezone.now)
    input_file = models.ForeignKey(ArchiveModel, on_delete=models.SET_NULL, null=True)
    task_id = models.TextField(default='task_id', max_length=1024)
    status = models.CharField(default='PENDING', max_length=256)
    task_type = models.CharField(default='unzip', max_length=256)
