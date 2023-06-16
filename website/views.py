from rest_framework.response import Response
from rest_framework import permissions, views, status
from pytube import YouTube
from website.models import (
    youtubeDownload,
    novelList,
    novelDetail,
)
from django.http import FileResponse
import json
from website.serializers import (
    youtubeDownloadSer,
    novelDetailSer,
)
import datetime
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
import re
import jieba
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
import pandas as pd
import requests,os,pytz
from website.tasks import get_id,test_task
from celery import Celery
import cv2
from django.core.cache import cache
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from logger import LogManager
import traceback

logger = LogManager().getLogger('view')

class test(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def post(self,request):
        image = cv2.imread('origan.jpg')

        image = Image.fromarray(image)
         # Convert the image to a binary format
        buffer = BytesIO()
        image.save(buffer, format='PNG')

        # binary
        image_binary = buffer.getvalue()
        input_image = Image.open(BytesIO(image_binary))
        # pil to png
        file = InMemoryUploadedFile(
            file=buffer,
            field_name=None,
            name=f'result.png',
            content_type='image/png',
            size=input_image.size,
            charset=None
        )
        print(file,type(file))
        

# class transferStyle(views.APIView):
#     permission_classes = (permissions.AllowAny,)
#     authentication_classes = []
#     def get(self,request):
#         transfer_style_objs = transferStyleModel.objects.all()
#         list_serializer = transferStyleSer(transfer_style_objs, many=True, context={"request": request}).data
#         return Response(list_serializer)


#     def post(self,request):
#         def load_file(image):
#             # np.array to Image.open
#             img_io = BytesIO()
#             image = Image.fromarray(image)
#             image.save(img_io, format='PNG')
#             img_io.seek(0)
#             image = Image.open(img_io)

#             max_dim = 512
#             factor = max_dim / max(image.size) # resize rate
#             '''
#             Image.NEAREST ：低质量
#             Image.BILINEAR：双线性
#             Image.BICUBIC ：三次样条插值
#             Image.ANTIALIAS：高质量
#             '''
#             image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.ANTIALIAS)
#             im_array = process_im.img_to_array(image) #to array
#             im_array = np.expand_dims(im_array, axis=0)  # adding extra axis to the array as to generate a
#             # batch of single image

#             return im_array

#         def show_im(img):
#             img=np.squeeze(img,axis=0) #squeeze array to drop batch axis #降一維
#             return np.uint8(img)

#         content_path_file = request.FILES['origan']
#         style_path_file = request.FILES['style']

#         content_path = content_path_file.read()
#         style_path = style_path_file.read()
#         # binary
#         content_path = np.array(Image.open(BytesIO(content_path)))
#         style_path = np.array(Image.open(BytesIO(style_path)))

#         content = load_file(content_path)
#         style = load_file(style_path)

#         def img_preprocess(img_path):
#             image=load_file(img_path)
#             img=tf.keras.applications.vgg19.preprocess_input(image) #to array
#             return img


#         def deprocess_img(processed_img):
#             x = processed_img.copy()
#             if len(x.shape) == 4:
#                 x = np.squeeze(x, 0) #down 1 ndim
#             assert len(x.shape) == 3  # assert 用於判斷一個表達式，若無滿足該表達式的條件，則直接觸發異常狀態，而不會接續執行後續的程式碼

#             # 設定RGB顏色的中心點 (Remove zero-center by mean pixel)
#             x[:, :, 0] += 103.939
#             x[:, :, 1] += 116.779
#             x[:, :, 2] += 123.68

#             # 'BGR'->'RGB'
#             x = x[:, :, ::-1]  # converting BGR to RGB channel
#             x = np.clip(x, 0, 255).astype('uint8') #min 0 max 255

#             return x

#         content_layers = ['block5_conv2']
#         style_layers = ['block1_conv1',
#                         'block2_conv1',
#                         'block3_conv1',
#                         'block4_conv1',
#                         'block5_conv1']
#         number_content=len(content_layers)
#         number_style =len(style_layers)


#         def get_model():
#             vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
#             vgg.trainable = False
#             content_output = [vgg.get_layer(layer).output for layer in content_layers]
#             style_output = [vgg.get_layer(layer).output for layer in style_layers]
#             model_output = style_output + content_output
#             return models.Model(vgg.input, model_output)

#         def get_content_loss(noise,target):
#             loss = tf.reduce_mean(tf.square(noise-target))
#             return loss

#         # 計算 風格 loss 的 gram matrix
#         def gram_matrix(tensor):
#             channels=int(tensor.shape[-1])
#             vector=tf.reshape(tensor,[-1,channels]) #x,[-1,channels] -> [[1,2,3],[4,5,6]]
#             n=tf.shape(vector)[0]
#             gram_matrix=tf.matmul(vector,vector,transpose_a=True)
#             '''
#             https://blog.csdn.net/qq_37591637/article/details/103473179
#             矩正相乘
#             [1 2 3 * [0 0 1   = [1*0+2*1+3*3 , 1*0+2*3+3*3,..
#             4 5 6]    1 3 2      4*0+5*1+6*3 ....              ]
#                     3 3 4] 
#             '''
#             return gram_matrix/tf.cast(n,tf.float32) #型別轉換

#         def get_style_loss(noise,target):
#             gram_noise=gram_matrix(noise)
#             #gram_target=gram_matrix(target)
#             loss=tf.reduce_mean(tf.square(target-gram_noise)) #計算reduce_mean所有這些浮點數的平均值。
#             return loss


#         def get_features(model, content_path, style_path):
#             content_img = img_preprocess(content_path)
#             style_image = img_preprocess(style_path)

#             content_output = model(content_img)
#             style_output = model(style_image)

#             content_feature = [layer[0] for layer in content_output[number_style:]]
#             style_feature = [layer[0] for layer in style_output[:number_style]]
#             return content_feature, style_feature


#         def compute_loss(model, loss_weights, image, gram_style_features, content_features):
#             style_weight, content_weight = loss_weights  # style weight and content weight are user given parameters
#             # that define what percentage of content and/or style will be preserved in the generated image

#             output = model(image)
#             content_loss = 0
#             style_loss = 0

#             noise_style_features = output[:number_style]
#             noise_content_feature = output[number_style:]

#             weight_per_layer = 1.0 / float(number_style)
#             for a, b in zip(gram_style_features, noise_style_features):
#                 style_loss += weight_per_layer * get_style_loss(b[0], a)

#             weight_per_layer = 1.0 / float(number_content)
#             for a, b in zip(noise_content_feature, content_features):
#                 content_loss += weight_per_layer * get_content_loss(a[0], b)

#             style_loss *= style_weight
#             content_loss *= content_weight

#             total_loss = content_loss + style_loss

#             return total_loss, style_loss, content_loss


#         def compute_grads(dictionary):
#             with tf.GradientTape() as tape:
#                 all_loss = compute_loss(**dictionary)

#             total_loss = all_loss[0]
#             return tape.gradient(total_loss, dictionary['image']), all_loss


#         def run_style_transfer(content_path, style_path, epochs=500, content_weight=1e3, style_weight=1e-2):
#             model = get_model()

#             for layer in model.layers:
#                 layer.trainable = False

#             content_feature, style_feature = get_features(model, content_path, style_path)
#             style_gram_matrix = [gram_matrix(feature) for feature in style_feature]

#             noise = img_preprocess(content_path)
#             noise = tf.Variable(noise, dtype=tf.float32)

#             optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1) #模糊因子

#             best_loss, best_img = float('inf'), None

#             loss_weights = (style_weight, content_weight)
#             dictionary = {'model': model,
#                         'loss_weights': loss_weights,
#                         'image': noise,
#                         'gram_style_features': style_gram_matrix,
#                         'content_features': content_feature}

#             norm_means = np.array([103.939, 116.779, 123.68])
#             min_vals = -norm_means
#             max_vals = 255 - norm_means

#             imgs = []
#             '''
#             即compute_gradients和apply_gradients，
#             前者用於計算梯度，
#             後者用於使用計算得到的梯度來更新對應的variable。下面對這兩個函數做具體介紹。
#             '''
#             for i in range(epochs):
#                 grad, all_loss = compute_grads(dictionary)
#                 total_loss, style_loss, content_loss = all_loss
#                 optimizer.apply_gradients([(grad, noise)])
#                 '''
#                 tf.clip_by_value的用法tf.clip_by_value(A, min, max)：
#                 輸入一個張量A，把A中的每一個元素的值都壓縮在min和max之間。
#                 小於min的讓它等於min，大於max的元素的值等於max
#                 '''
#                 clipped = tf.clip_by_value(noise, min_vals, max_vals)
#                 '''
#                 https://medium.com/ai-blog-tw/tensorflow-%E4%BB%80%E9%BA%BC%E6%98%AFassign-operator-%E4%BB%A5tf-assign%E5%AF%A6%E4%BD%9Ccounter-184479257531
#                 assign 剛剛上面的寫法問題在於，不管你跑幾次，
#                 var這個tf.Variable的值都從未更新過，所以需要使用tf.assign指定新的數值
#                 '''
#                 noise.assign(clipped)

#                 if total_loss < best_loss:
#                     best_loss = total_loss
#                     best_img = deprocess_img(noise.numpy())

#                 # for visualization

#                 if i % 5 == 0:
#                     plot_img = noise.numpy()
#                     plot_img = deprocess_img(plot_img)
#                     imgs.append(plot_img)
#                     IPython.display.clear_output(wait=True)
#                     IPython.display.display_png(Image.fromarray(plot_img))
#                     print('Epoch: {}'.format(i))
#                     print('Total loss: {:.4e}, '
#                         'style loss: {:.4e}, '
#                         'content loss: {:.4e}, '.format(total_loss, style_loss, content_loss))

#             IPython.display.clear_output(wait=True)

#             return best_img, best_loss, imgs
        
#         def np_convert_inMemoryUpFile(image):
#             # Step 1: Convert the NumPy array to an image
#             image = Image.fromarray(image)
#             # Convert the image to a binary format
#             buffer = BytesIO()
#             image.save(buffer, format='PNG')

#             # binary
#             image_binary = buffer.getvalue()
#             input_image = Image.open(BytesIO(image_binary))
#             # pil to png
#             result = InMemoryUploadedFile(
#                 file=buffer,
#                 field_name=None,
#                 name=f'result.png',
#                 content_type='image/png',
#                 size=input_image.size,
#                 charset=None
#             )
#             return result

#         best, best_loss,image = run_style_transfer(content_path,
#                                             style_path, epochs=20)
#         content = show_im(content)
#         style = show_im(style)
#         result = np_convert_inMemoryUpFile(best)

#         if 'origan' in request.data and 'style' in request.data:
#             defaults = dict()
#             defaults['name'] = str(content_path_file)
#             defaults['origan'] = content_path_file
#             defaults['style'] = style_path_file
#             defaults['result'] = result
#             transferStyleModel.objects.update_or_create(
#                 pk = request.data.get('id'),
#                 defaults=defaults
#             )
        
#         # Convert the NumPy array to an image
#         image = Image.fromarray(best)
#         # Create an in-memory file-like object
#         file_buffer = BytesIO()
#         # Save the image into the file-like object
#         image.save(file_buffer, format='PNG')
#         response = HttpResponse(content_type="image/png")
#         response["Content-Disposition"] = "attachment; filename=image.png"
#         # Set the file buffer as the response content
#         response.content = file_buffer.getvalue()
#         return response


class trackCap(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def get(self,request):
        try:
            bs = cv2.bgsegm.createBackgroundSubtractorGMG()
            cap = cv2.VideoCapture(0)
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while True:
                ret,frame = cap.read()
                frame = cv2.flip(frame,1) #移動同邊 

                gray = bs.apply(frame)
                thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)[1]
                erode = cv2.erode(thresh,None,iterations=2)
                dilate = cv2.dilate(erode,None,iterations=2)
                
                cnts,hierarchy = cv2.findContours(
                    dilate,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for c in cnts:
                    if cv2.contourArea(c) > 30:
                        #畫出輪廓
                        # cv2.drawContours(frame,cnts,-1,(0,255,255),2)
                        (x,y,w,h) = cv2.boundingRect(c)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        
                frame2 = cv2.bitwise_or(frame,frame,mask=dilate)
                frame = cv2.hconcat([frame,frame2]) #水平擴展
                cv2.imshow('frame',frame)
                if cv2.waitKey(10) == ord('q'):
                    break

            cv2.destroyAllWindows()
            return Response({"success": True, "message": "cap is close"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"success": False, "message": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class youBike(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def get(self,request):
        try:
            '''
            sno(站點代號)、sna(中文場站名稱)、tot(場站總停車格)、sbi(可借車位數)、
            sarea(中文場站區域)、mday(資料更新時間)、lat(緯度)、lng(經度)、
            ar(中文地址)、sareaen(英文場站區域)、snaen(英文場站名稱)、aren(英文地址)、
            bemp(可還空位數)、act(場站是否暫停營運)
            '''
            logger.info('youbike get method strat...')
            url = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
            res = requests.get(url).json()
            df = pd.DataFrame(res)[['sna','updateTime','tot','sbi','bemp','lat','lng','sarea','ar']]
            #模糊比對
            key_word = request.GET.get('keyWord', None)
            if key_word:
                df = df[(df.sna.str.contains(key_word))|(df.ar.str.contains(key_word))|(df.sarea.str.contains(key_word))]

            df.sort_values(by=['sbi','bemp'],ascending=[False,False],axis=0,inplace=True)

            df.rename(columns={
                'sna':'站點','updateTime':'更新時間','tot':'場站總停車格','sbi':'可借車位數','bemp':'可還空位數',
                'lat':"緯度",'lng':'經度','sarea':'場站區域','ar':'地址'},inplace=True)
            logger.info('youbike get method finish...')
            return Response(df.T.to_dict().values(),status=200)
        except Exception as e:
            logger.error(traceback.format_exc())
            return Response({'message':'error'},status=400)

# Create your views here.
class downloadYoutube(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def get(self,request):
        start = request.GET.get('start', None)
        end = request.GET.get('end', None)

        youtube_objs = youtubeDownload.objects.all()
        if start and (start != json.dumps(None)):
            start = datetime.datetime.strptime(start,"%Y-%m-%d")
            youtube_objs = youtube_objs.filter(created_at__gte=start)
        if end and (end != json.dumps(None)):
            start = datetime.datetime.strptime(end,"%Y-%m-%d")
            youtube_objs = youtube_objs.filter(created_at__lte=end)
        
        list_serializer = youtubeDownloadSer(youtube_objs, many=True, context={"request": request}).data
        return Response(list_serializer)

    def post(self, request):
        defaults = dict()
    
        fileName = request.data.get('fileName')
        url = request.data.get('url')

        defaults['ip_address'] = request.META['REMOTE_ADDR']
        defaults['url'] = url
        if fileName:
            defaults['file_name'] = fileName
        
        youtubeDownload.objects.update_or_create(
            pk=request.data.get('id'),
            defaults=defaults
        )
       
        return FileResponse(open(YouTube(url).streams.filter().get_highest_resolution().download(skip_existing=True),'rb'))

class downloadNovelWordCloud(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def get(self,request):
        name = request.GET.get('name', None)
        novelId = request.GET.get('novelId', None)
        novels = novelDetail.objects.all()

        if name and (name != json.dumps(None)):
            novels = novels.filter(novel__name__contains=name)
        if novelId and (novelId != json.dumps(None)):
            novels = novels.filter(id=novelId)

        list_serializer = novelDetailSer(novels, many=True, context={"request": request}).data
        return Response(list_serializer)

    def post(self,request):
        ua = UserAgent()
        url = request.data.get('url')
        user_agent = {"User-Agent":f"{ua.chrome}"}
        res = requests.get(url,headers=user_agent)
        
        logger.info('novel post start....')

        soup = BeautifulSoup(res.text,'lxml')
        name = soup.select_one(".name").text

        defaults = dict()
        defaults['name'] = name
        obj,created = novelList.objects.update_or_create(
            pk=request.data.get('id'),
            defaults=defaults
        )
        #載入停用字
        stopwords = set([stopword.strip() for stopword in \
            open("stopword.txt", "r", encoding="utf-8").readlines()])

        try:
            chapter = str(url.split('/')[-1].split('.html')[0])
            content = soup.select_one('#content').text

            # 只取文字
            content = "".join(re.findall(r'\w+',content))
            # jieba切割
            jieba_words = "|".join(jieba.cut(content))
            # 文字雲套件
            wc = WordCloud(width=400,height=400,background_color="white", stopwords=stopwords, font_path="kaiu.ttf").generate(jieba_words)

            image = wc.to_image()
            # Convert the image to a binary format
            buffer = BytesIO()
            image.save(buffer, format='PNG')

            # binary
            image_binary = buffer.getvalue()
            input_image = Image.open(BytesIO(image_binary))
            # pil to png
            file = InMemoryUploadedFile(
                file=buffer,
                field_name=None,
                name=f'{chapter}.png',
                content_type='image/png',
                size=input_image.size,
                charset=None
            )
            
            defaults = dict()
            defaults['content'] = content
            defaults['file_path'] = file
            novelDetail.objects.update_or_create(
                novel_id=obj.id,
                chapter=chapter,
                defaults=defaults
            )
            logger.info('novel post finish....')
        except Exception as e:
            logger.error(traceback.format_exc())
            return Response({"message":"error"},status=400)
        return Response({"message":True},status=201)
    def delete(self,request):
        try:
            ids = request.data.get('ids')
            novelDetail.objects.filter(id__in=ids).delete()
            return Response({'message':'delete success'},status=200)
        except:
            logger.error(traceback.format_exc())
            return Response({'message':'delete false'},status=200)
        

class testAsyncTask(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def post(self,request):
        id = request.data.get('id')

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
        app = Celery("mysite")
        app.config_from_object("django.conf:settings", namespace="CELERY")
        app.autodiscover_tasks()

        local_timezone = pytz.timezone("Asia/Taipei")
        now = datetime.datetime.now()
        # 1秒後執行
        exec_time = now + datetime.timedelta(seconds=1)
        # 利用 pytz 進行轉換
        exec_time = local_timezone.localize(exec_time)
    
        get_id.apply_async(args=(id,), eta=exec_time)

        return Response( {"success": True, "message": "執行成功"}, status=status.HTTP_200_OK)
    

class redisTaskTest(views.APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def get(self,request):
        # task + redis
        now = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d-%H:%M")[:-1]+'0'
        if cache.has_key(now):
            youbike_data = cache.get(now)
            return Response(youbike_data.values(),status=200)
        else:
            return Response('error',status=400)