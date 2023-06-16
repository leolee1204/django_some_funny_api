from django.db import models
from datetime import datetime
import os
import uuid
# Create your models here.

def get_wordcloud_file_path(instance,name):
    return f'wordCloud/{instance.novel.name}/{name}'

def get_origan_file_path(instance,name):
    return f'transferStyle/{instance.name}/origan.jpg'
def get_style_file_path(instance,name):
    return f'transferStyle/{instance.name}/style.jpg'
def get_result_file_path(instance,name):
    return f'transferStyle/{instance.name}/result.jpg'

class youtubeDownload(models.Model):
    id = models.AutoField(primary_key=True)
    ip_address = models.CharField(max_length=255, null=True, blank=True)
    url = models.URLField(max_length = 500,null=True, blank=True)
    file_name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(default=datetime.now, null=False)

class novelList(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255,null=True,blank=True)
    created_at = models.DateTimeField(default=datetime.now, null=False)

class novelDetail(models.Model):
    id = models.AutoField(primary_key=True)
    novel = models.ForeignKey(
        novelList,on_delete=models.PROTECT,
        to_field='id',
        default=None,
        blank=True,
        null=True,
        related_name='novelDetail'
    )
    chapter = models.IntegerField(null=True,blank=True)
    content = models.TextField(blank=True, null=True)
    file_path = models.FileField(upload_to=get_wordcloud_file_path,null=True, blank=True, default=None)
    created_at = models.DateTimeField(auto_now_add = True,editable=False)

class transferStyleModel(models.Model):
     id = models.AutoField(primary_key=True)
     name = models.CharField(max_length=255,null=True,blank=True)
     origan = models.FileField(upload_to=get_origan_file_path,null=True, blank=True, default=None)
     style = models.FileField(upload_to=get_style_file_path,null=True, blank=True, default=None)
     result = models.FileField(upload_to=get_result_file_path,null=True, blank=True, default=None)
     created_at = models.DateTimeField(auto_now_add = True,editable=False)