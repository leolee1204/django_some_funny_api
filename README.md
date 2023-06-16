# project001

### api/youtube/download post
{
    "fileName":"name",
    "url":"youtube url"
}
![](https://github.com/leolee1204/project001/blob/2a5c46b97833e23b4661d8647b4acae4222c5221/images/youtube_post.png)

### api/youtube/download get
start >= date
end <= date
start , end in params
![](https://github.com/leolee1204/project001/blob/2a5c46b97833e23b4661d8647b4acae4222c5221/images/youtube_get.png)

### api/novel/download post
https://big5.quanben.io/
choose novel and chapter ,download product wordCloud

choose novel and post url
ex:https://big5.quanben.io/n/chaojinongyebazhu/2.html
![](https://github.com/leolee1204/project001/blob/a67fc2ba0a980e5582fd8621642e11ba2e528c84/images/novel_post.png)

#### picture
![](https://github.com/leolee1204/project001/blob/2a5c46b97833e23b4661d8647b4acae4222c5221/media/wordCloud/%E9%AC%A5%E7%BE%85%E5%A4%A7%E9%99%B8/1.png)

### api/novel/download get
name , novel filter in params
name -> 模糊比對(str)
novel : int
![](https://github.com/leolee1204/project001/blob/2a5c46b97833e23b4661d8647b4acae4222c5221/images/novel_get.png)

### api/youbike get
keyWord filter in params
keyWord -> 模糊比對(str)
filter 站點 | 場站區域 | 地址
sort_by("可借車位數","可還空位數")
![](https://github.com/leolee1204/project001/blob/020db3cda9ed421db65333ec4e1f4560f49445b2/images/youbike_get.png)
