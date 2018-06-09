from urllib import request,parse
import string
import random
import io
from PIL import Image
import matplotlib.pyplot as plt

img_url = 'http://jwxt.dgut.edu.cn/dglgjw/cas/genValidateCode?'
valid_url = 'http://jwxt.dgut.edu.cn/dglgjw/public/dykb.GS1.jsp?kblx=jskb'
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
    'Cookie':'_xsrf=2|1fd9ce14|28536d81340e403377c1c9e092947f09|1522744930; JWC_SERVERID=jwc1; JSESSIONID=F9B9366808663F32214DD38972A98932; JWXT_HA=ha11'
}

def retImgFromInternet():
    req = request.Request(url=img_url, headers=headers)
    image_bytes = request.urlopen(req).read()
    data_stream = io.BytesIO(image_bytes)
    pil_image = Image.open(data_stream)
    return pil_image

def upload_get_result(predict):
    dict = {'randnumber':str(predict)}
    data = bytes(parse.urlencode(dict), encoding="utf8")
    req = request.Request(url=valid_url,data=data,headers=headers,method='post')
    text = request.urlopen(req).read()
    return '验证码正确' if len(text)==2045 else '验证码错误'

def download_jpg():
    for i in range(60):
        imgName = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        request.urlretrieve(img_url, '../test_img/'+imgName+'.jpg')

