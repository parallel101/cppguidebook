from collections import namedtuple
import os
import requests
import time
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

class User(namedtuple('User', ['name', 'avatar', 'all_sum_amount'])):
    pass

manual_sponsors = [
    User('等疾风', 'https://i0.hdslb.com/bfs/face/b658b5ca52f41e53321d04f978be6784ca6f8687.jpg', '1000.00'),
    User('只喝白开水', 'https://i2.hdslb.com/bfs/face/821b88a24c1319d1fb51b3854884e2f829855c75.jpg', '100.00'),
    User('包乾', '', '26.90'),
    User('柿柿如意', '', '20.00'),
    User('Starry', '', '100.00'),
    User('阿哲', '', '100.00'),
    User('Eureka', '', '20.00'),
    User('孙斌', '', '200.00'),
    User('nullptr', 'https://i0.hdslb.com/bfs/face/effa1ec9bb0f5d09ed415da75129aca9d16092ac.jpg', '23.30'),
    User('Fred Song', '', '25.00'),
    User('**振', '', '20.00'),
]

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
            res.append(User(user['user']['name'], user['user']['avatar'], user['all_sum_amount']))
        if i >= n:
            break
        i += 1
    return res

def afd_gen_thank_list():
    sponsors = afd_query_sponsors()
    sponsors += manual_sponsors
    max_x = 30
    max_y = 30
    limit_y = 600
    max_max_y = max_y
    for user in sponsors:
        max_y += 100
        if max_y + 10 >= limit_y:
            max_max_y = max(max_max_y, max_y)
            max_y = 30
            max_x += 400
    max_max_y = max(max_max_y, max_y)
    max_max_x = max_x + 400
    max_y += 10
    img = Image.new('RGB', (max_max_x, max_max_y), color='#19242e')
    x = 30
    y = 30
    total = 0
    for user in sponsors:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('/usr/share/fonts/noto-cjk/NotoSansCJK-Medium.ttc', size=20)
        if user.avatar:
            avatar = Image.open(BytesIO(requests.get(user.avatar).content))
        elif os.path.exists(f'/home/bate/下载/wx-{user.name.replace('/', '|')}.png'):
            avatar = Image.open(f'/home/bate/下载/wx-{user.name.replace('/', '|')}.png')
        else:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            avatar = Image.open(os.path.join(this_dir, '../docs/img/favicon.ico'))
        avatar = avatar.resize((80, 80))
        img.paste(avatar, (x, y))
        draw.text((x + 100, y), f'{user.name}', fill='white', font=font)
        draw.text((x + 100, y + 30), f'￥{user.all_sum_amount}', fill='#aaaaaa', font=font)
        print(f'{user.name} ￥{user.all_sum_amount}')
        total += float(user.all_sum_amount)
        y += 100
        if y + 10 >= limit_y:
            y = 30
            x += 400
    print(total)
    return img

img = afd_gen_thank_list()
file = 'docs/img/thanks.png'
img.save(file)
img.show()
