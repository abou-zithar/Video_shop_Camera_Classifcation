from django.db import models


class VideoUpload(models.Model):
    video = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Video {self.id} - {self.uploaded_at}'
