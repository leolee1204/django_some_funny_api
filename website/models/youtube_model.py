from django.db import models
from datetime import datetime

class youtubeDownload(models.Model):
    id = models.AutoField(primary_key=True)
    ip_address = models.CharField(max_length=255, null=True, blank=True)
    url = models.URLField(max_length = 500,null=True, blank=True)
    file_name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(default=datetime.now, null=False)