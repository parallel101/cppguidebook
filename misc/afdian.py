import os
import requests
import time
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def afd_query(which, **params):
    user_id = '6256dedc1af911eebf8152540025c377'
    token = os.environ['AFDIAN_TOKEN']
    ts = int(time.time())
    params = json.dumps(params)
    sign = hashlib.md5(f'{token}params{params}ts{ts}user_id{user_id}'.encode('utf-8')).hexdigest()
    res = requests.get(f'https://afdian.net/api/open/{which}', params={
        'user_id': user_id,
        'params': params,
        'ts': ts,
        'sign': sign,
    }).json()
    assert res['ec'] == 200, res
    return res['data']

def afd_query_sponsors():
    i = 1
    res = []
    while True:
        page = afd_query('query-sponsor', page=i)
        n = page['total_page']
        for user in page['list']:
            res.append(user)
        if i >= n:
            break
        i += 1
    return res

def afd_gen_thank_list():
    sponsors = list(reversed(afd_query_sponsors()))
    max_y = 30
    for user in sponsors:
        max_y += 100
    max_y += 10
    img = Image.new('RGB', (800, max_y), color='#19242e')
    x = 30
    y = 30
    for user in sponsors:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('/usr/share/fonts/noto-cjk/NotoSansCJK-Medium.ttc', size=20)
        avatar = Image.open(BytesIO(requests.get(user['user']['avatar']).content))
        avatar = avatar.resize((80, 80))
        img.paste(avatar, (x, y))
        draw.text((x + 100, y), f'{user['user']['name']}', fill='white', font=font)
        draw.text((x + 100, y + 30), f'￥{user['all_sum_amount']}', fill='#aaaaaa', font=font)
        print(f'{user['user']['name']} ￥{user['all_sum_amount']}')
        print(user)
        y += 100
    return img

img = afd_gen_thank_list()
file = 'docs/img/thanks.png'
img.save(file)
