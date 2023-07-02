# project001

下載yotube影片，可以指定下載位置，操作如下
### api/youtube/download post
{
    "fileName":"name",
    "url":"youtube url"
}
![](https://github.com/leolee1204/project001/blob/f94213fa482b9acae949a51322317961383a9a06/images/youtube_post.png)


### api/youtube/download get
* start >= date
* end <= date
* start , end in params
* 也可以不帶params 直接get相關資料
* 
![](https://github.com/leolee1204/project001/blob/cdfe41e0238bd982ba6be80567a021cb0324f1ee/images/youtube_get.png)

### api/novel/download post

* 爬取網址，生成文字雲，再從get method去觀看
* https://big5.quanben.io/
* choose novel and chapter ,download product wordCloud
* choose novel and post url
* ex:https://big5.quanben.io/n/chaojinongyebazhu/2.html
![](https://github.com/leolee1204/project001/blob/f94213fa482b9acae949a51322317961383a9a06/images/novel_post.png)

### api/novel/download get
* name , id filter in params
* name -> 模糊比對(str)
* id : int
* 創建越後面的時間，回傳資料會至頂
![](https://github.com/leolee1204/project001/blob/f94213fa482b9acae949a51322317961383a9a06/images/novel_get.png)

### api/novel/download delete
* {"ids":[241,240]}
* 透過傳送json ids，將list內相關ids刪除
![](https://github.com/leolee1204/project001/blob/f94213fa482b9acae949a51322317961383a9a06/images/novel_delete.png)

### api/youbike get
* keyWord filter in params
* keyWord -> 模糊比對(str)
* filter 站點 | 場站區域 | 地址
* sort_by("可借車位數","可還空位數")
* 生成地理區域圖表，可以伸縮html圖形，可以count附近站點
* 可租借車位 or 可還車車位 < 5 就會呈現紅色，否則綠色
* 點擊所選取的站點，可以觀看站點名稱,可租借車位,可還車車位,更新時間
* 資料即時爬取opendata youbike2.0
* 輸入 https://projecct001.onrender.com/api/youbike
* 後面可帶 ?keyWord=xxx 模糊比對 或 不帶參數也可以
![](https://github.com/leolee1204/project001/blob/cdfe41e0238bd982ba6be80567a021cb0324f1ee/images/youbike_get.png)

### api/stock get
* name 找尋美股股票名稱 example AAPL or 2330.TW(台灣)
* startDate 起始時間 預設 前一年1月1日
* endDate 終止時間 預設 今天
* 生成plotly頁面 上面為蠟燭圖 下面為交易量
![](https://github.com/leolee1204/project001/blob/9d1fb990fe2a47c7ef8b013f32c25ab7d8617a23/images/stock_get.png)
