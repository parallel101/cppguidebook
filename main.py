def define_env(env):
    env.variables.icon = dict(
        fun='<img src="../img/awesomeface.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        tip='<img src="../img/bulb.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        story='<img src="../img/book.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        warn='<img src="../img/warning.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        detail='<img src="../img/question.png" height="30px" width="auto" style="margin: 0; border: none"/>',
    )
    env.variables.icon2 = dict(
        fun='<img src="./img/awesomeface.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        tip='<img src="./img/bulb.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        story='<img src="./img/book.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        warn='<img src="./img/warning.png" height="30px" width="auto" style="margin: 0; border: none"/>',
        detail='<img src="./img/question.png" height="30px" width="auto" style="margin: 0; border: none"/>',
    )
    import datetime
    env.variables.build_date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y年%m月%d日 %H:%M:%S (%Z)')
