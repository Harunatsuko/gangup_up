from django.db import models
from django.utils.timezone import timezone


class ArchiveModel(models.Model):
    file = models.FileField


class TaskModel(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    task_id = models.IntegerField()
