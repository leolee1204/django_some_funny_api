import traceback, datetime, json, base64
from functools import wraps
from website.models.sys_model import Sys_Event_Log
from com.mts.logger import LogManager
from website.models.sys_model import Sys_Event_Log,Sys_Event_Code,Sys_Subject
from website.models.users_model import Users
from website.serializers.sys_ser import SysEventCodeSerializer, SysSubjectSerializer

logger = LogManager().getLogger('view')
sys_event_code_data = Sys_Event_Code.objects.values_list('event_id', flat=True)
sys_subject_data = Sys_Subject.objects.values_list('subject_id', flat=True)

# event_id: view_func執行前和後的log紀錄，A:前A後IorE / I:前I / E:不合理 / R:讀取(查詢)
# keys: ["session.key", "path.key", "param.key", "data,key"]
    # path : kwargs[key], path路徑
    # data : request.data[key], post內容
    # param : request.GET[key], ?後面的attribute
    # session : request.session[key], 取seesion內容, 如user_id
# message1: 使用%s、%f等字串格式自行handle
def logtodb(event_id:str='999999',subject_id:str='999999',
            message1:str=None, message2:str=None, keys:dict=None):
    def my_decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            keep_message1 = ''; keep_message2 = ''
            origin_event_id = event_id ; origin_subject_id = subject_id
            response = None
            # 統一時間戳記
            # log_at = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S.%f')
            log_at = datetime.datetime.now()
            # 如果是登入行為，則user_id會是None, 從username去取(post data)
            if '/login' in request.get_full_path():
                user_id = None
                # if 'username' in [i[0] for i in request.data]:
                #     user_id = Users.objects.get(auth_user__username=request.data['username']).id
            else:
                user_id = request.session['user_id'] if 'user_id' in request.session else None
            auth_user_id = request.session['auth_user_id'] if 'auth_user_id' in request.session else None
            try:
                # ------
                # 動作執行前寫入sys_event_log, 如果發生問題不應該影響動作繼續執行
                # ------
                my_param = None ; info1 = None
                try:
                    keep_message2 = message2
                    keep_event_id = event_id if (event_id in sys_event_code_data and event_id[0] != 'E') else '999999'
                    keep_subject_id = subject_id if subject_id in sys_subject_data else '999999'
                    my_param = tuple()
                    if keys is not None:
                        for i in keys:
                            try:
                                if i[0] == 'session':
                                    my_param += (request.session[i[1]],)
                                elif i[0] == 'param':
                                    my_param += (request.GET[i[1]],)
                                elif i[0] == 'data':
                                    my_param += (request.data[i[1]],)
                                else:
                                    my_param += (kwargs[i[1]],)
                            except:
                                my_param += ('',)                       

                        # my_param = tuple((request.session[i[1]] if i[0]=='session' 
                        #             else request.GET[i[1]] if i[0]=='param' 
                        #             else request.data[i[1]] if i[0]=='data'
                        #             else kwargs[i[1]] for i in keys))
                    try:
                        keep_message1 = message1 %my_param if my_param else message1
                    except:
                        keep_message1 = message1                        
                    # info1 = None if my_param else str(keys)
                    level = keep_event_id[0] if keep_event_id[0] in ['I','A','E', 'R'] else '-'
                    
                    sel_obj = Sys_Event_Log(log_at=log_at, level=level, message1=keep_message1,
                                        message2=keep_message2, info1=info1,
                                        event_id=Sys_Event_Code.objects.get(pk=keep_event_id),
                                        subject_id=Sys_Subject.objects.get(pk=keep_subject_id), 
                                        user_id=Users.objects.get(pk=user_id) if user_id else None)
                    sel_obj.save()
                except:
                    info1 = 'event_id:%s, subject_id:%s, key:%s' %(origin_event_id, origin_subject_id ,str(keys))
                    sel_obj = Sys_Event_Log(log_at=log_at, level='E', message1=message1,
                                        message2=message2, info1=info1,
                                        event_id=Sys_Event_Code.objects.get(pk='999999'),
                                        subject_id=Sys_Subject.objects.get(pk='999999'), 
                                        user_id=Users.objects.get(pk=user_id) if user_id else None)
                    sel_obj.save()
                    logger.error('sys_event_log 參數解析異常. 將訊息寫到info1')
                    logger.error(traceback.format_exc())
                # ------
                # 執行view
                # ------
                response = view_func(request, *args, **kwargs)
                if str(response.status_code)[0] in ['4','5']:
                    raise ValueError('Response:4xx/5xx請求失敗,請參考sys_event_log table.')
                
                # ------
                # 如果是登入行為則在執行完view才會將user_id寫入session
                # 如果user_id仍然是None則可能原因是什麼???(瀏覽器禁用cookie?)
                # ------
                idvalue = json.loads(base64.b64decode(request.headers['X-Userinfo'])) \
                          if 'X-Userinfo' in request.headers else None
                user_id = request.session['user_id'] if 'user_id' in request.session \
                          else idvalue['ext']['id'] if 'ext' in idvalue else None
                if user_id is None:
                    raise ValueError('user_id 不存在, 未知的錯誤.')

                # ------
                # 執行成功,新增寫入sys_event_log , 動作執行後訊息
                # event_id為"A"開頭才有完成log
                # ------
                try:
                    if keep_event_id[0] == 'A':
                        check_event_id = 'I' + keep_event_id[1:]
                        check_event_id = check_event_id if check_event_id in sys_event_code_data else '999999'
                        sel_obj = Sys_Event_Log(log_at=log_at, level='I', message1=keep_message1,
                                message2=keep_message2, info1=info1,
                                event_id=Sys_Event_Code.objects.get(pk=check_event_id),
                                subject_id=Sys_Subject.objects.get(pk=keep_subject_id), 
                                user_id=Users.objects.get(pk=user_id) if user_id else None)
                        sel_obj.save()
                except:
                    sel_obj = Sys_Event_Log(log_at=log_at, level='E', message1=message1,
                                message2=message2, info1=info1,
                                event_id=Sys_Event_Code.objects.get(pk='999999'),
                                subject_id=Sys_Subject.objects.get(pk='999999'), 
                                user_id=Users.objects.get(pk=user_id) if user_id else None)
                    sel_obj.save()
                    logger.error('sys_event_log 參數解析異常. 將訊息寫到info1')
                    logger.error(traceback.format_exc())

            except:
                # 當view_func發生except或status_code='4xx'
                # TODO 寫入sys_event_log, 是否回500?
                check_event_id = 'E' + keep_event_id[1:]
                check_event_id = check_event_id if check_event_id in sys_event_code_data else '999999'
                sel_obj = Sys_Event_Log(log_at=log_at, level='E', message1=keep_message1,
                                        message2=keep_message2, info1='失敗原因:api執行異常或log寫入失敗. 相關參數:%s' %info1,
                                        event_id=Sys_Event_Code.objects.get(pk=check_event_id),
                                        subject_id=Sys_Subject.objects.get(pk=keep_subject_id), 
                                        user_id=Users.objects.get(pk=user_id) if user_id else None)
                sel_obj.save()
                logger.error('發生錯誤,寫錯誤代碼到sys_event_log')
                logger.error(traceback.format_exc())
            return response
        return wrapper
    return my_decorator
