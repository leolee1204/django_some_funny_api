from celery import Celery
from celery import shared_task
import datetime
import requests
import pandas as pd
import json
from django.core.cache import cache
from com.mts.logger import LogManager

app = Celery("django_practice")

@shared_task()
def test_task():
    return 'finish:' + str(datetime.datetime.now())

@shared_task()
def get_id(id):
    return 'finish:' + str(id)

@shared_task
def get_youbike():
    logger = LogManager().getLogger('youbike tasks')
    now = datetime.datetime.now()
    # 每五分鐘當一個key
    key = now.replace(minute=int(now.minute)//5,second=0,microsecond=0)
    '''
    sno(站點代號)、sna(中文場站名稱)、tot(場站總停車格)、sbi(可借車位數)、
    sarea(中文場站區域)、mday(資料更新時間)、lat(緯度)、lng(經度)、
    ar(中文地址)、sareaen(英文場站區域)、snaen(英文場站名稱)、aren(英文地址)、
    bemp(可還空位數)、act(場站是否暫停營運)
    '''
    logger.info('youbike task strat...')
    url = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
    res = requests.get(url).json()
    df = pd.DataFrame(res)[['sna','updateTime','tot','sbi','bemp','lat','lng','sarea','ar']]
    #模糊比對
    cache.set(df,60*5)
    logger.info('youbike task finish...')