import zipfile
from os import listdir
from os.path import isfile, join
from celery import shared_task
from .models import ArchiveModel
from .tracking import wrapper

@shared_task
def input_archive(input_path, output_path, model_id) -> bool:
    """
    Распаковать архив и присвоить к ArchiveModel путь к распакованным файлам, запустить трекинг
    :param input_path: путь к архиву
    :param output_path: путь куда распаковать
    :param model_id: айди модели ArchiveModel
    :return: status
    """
    try:
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
            archive_model = ArchiveModel.objects.get(pk=model_id)
            archive_model.unzipped_path = output_path
            archive_model.save()

            onlyfiles = [f for f in listdir(f"{output_path}/frames_rgb/") if
                         isfile(join(f"{output_path}/frames_rgb/", f))]
            files = [f"{output_path}/frames_rgb/" + frame for frame in
                     onlyfiles]
            files.sort()
            wrapper(files, archive_model.pk, archive_model.show_rgb, archive_model.show_tiff)

        return True
    except Exception as err:
        print(err)
        return False

