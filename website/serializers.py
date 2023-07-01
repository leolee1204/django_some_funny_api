from rest_framework import serializers
from website.models import (
    youtubeDownload,
    novelDetail,
)

class youtubeDownloadSer(serializers.ModelSerializer):
    class Meta:
        model = youtubeDownload
        fields = "__all__"

class novelDetailSer(serializers.ModelSerializer):
    name = serializers.ReadOnlyField(source='novel.name')
    class Meta:
        model = novelDetail
        exclude =('created_at',)
        extra_fields = ["name",]
