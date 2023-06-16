from celery import Celery
from celery import shared_task
import datetime
import requests
import pandas as pd
import json
from django.core.cache import cache

app = Celery("django_practice")

@shared_task()
def test_task():
    return 'finish:' + str(datetime.datetime.now())

@shared_task()
def get_id(id):
    return 'finish:' + str(id)

@shared_task
def get_youbike():
    now = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d-%H:%M")[:-1]+'0'
    '''
    sno(站點代號)、sna(中文場站名稱)、tot(場站總停車格)、sbi(可借車位數)、
    sarea(中文場站區域)、mday(資料更新時間)、lat(緯度)、lng(經度)、
    ar(中文地址)、sareaen(英文場站區域)、snaen(英文場站名稱)、aren(英文地址)、
    bemp(可還空位數)、act(場站是否暫停營運)
    '''
    url = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
    res = requests.get(url).json()
    df = pd.DataFrame(res)[['sna','updateTime','tot','sbi','bemp','lat','lng','sarea','ar']]
    
    df.sort_values(by=['sbi','bemp'],ascending=[False,False],axis=0,inplace=True)
    df.rename(columns={
        'sna':'站點','updateTime':'更新時間','tot':'場站總停車格','sbi':'可借車位數','bemp':'可還空位數',
        'lat':"緯度",'lng':'經度','sarea':'場站區域','ar':'地址'},inplace=True)

    result = df.T.to_dict()
    # 存1分鐘
    cache.set(now, result, 60 * 3)
    return now + "finish"