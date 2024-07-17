#set text(
  font: "Noto Serif CJK SC",
  size: 7pt,
)
#set page(
  paper: "a6",
  margin: (x: 1.8cm, y: 1.5cm),
  header: align(right, text(5pt)[
    Unicode 宇宙
  ]),
  numbering: "1",
)
#set par(justify: true)
#set heading(numbering: "1.")
#show "Unicode": name => box[
    #text(fill: rgb("#4d932e"))[*#name*]
]
#show "UTF-32": name => box[
    #text(fill: rgb("#4d932e"))[*#name*]
]
#show "UCS-4": name => box[
    #text(fill: rgb("#4d932e"))[*#name*]
]
#show "UTF-16": name => box[
    #text(fill: rgb("#1182ad"))[*#name*]
]
#show "UCS-2": name => box[
    #text(fill: rgb("#1f9e8c"))[*#name*]
]
#show "UTF-8": name => box[
    #text(fill: rgb("#bc8f08"))[*#name*]
]
#show "Latin-1": name => box[
    #text(fill: rgb("#8a6643"))[*#name*]
]
#show "ISO-8859-1": name => box[
    #text(fill: rgb("#8a6643"))[*#name*]
]
#show "ASCII": name => box[
    #text(fill: rgb("#be6a6a"))[*#name*]
]
#show "GB2312": name => box[
    #text(fill: rgb("#8c4ea4"))[*#name*]
]
#show "GBK": name => box[
    #text(fill: rgb("#6645b1"))[*#name*]
]
#show "GB18030": name => box[
    #text(fill: rgb("#7065eb"))[*#name*]
]
#show "ANSI": name => box[
    #text(fill: rgb("#d06dd1"))[*#name*]
]

#let fun = body => box[
    #box(image(
        "pic/awesomeface.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#cd9f0f"))[#body]
]
#let tip = body => box[
    #box(image(
        "pic/bulb.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#4f8b4f"))[#body]
]
#let warn = body => box[
    #box(image(
        "pic/warning.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#ed6c6c"))[#body]
]
#let story = body => box[
    #box(image(
        "pic/book.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#807340"))[#body]
]
#let detail = body => box[
    #box(image(
        "pic/question.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#8080ad"))[#body]
]

#let comment = name => ""
#show table.cell.where(x: 0): strong
#let codetab = (u, v, a, b, n) => table(
    columns: int((a.len() + n - 1) / n) + 1,
    inset: 3pt,
    align: horizon,
    ..range(0, n).map(i =>
        (
            [#u], ..a.slice(int((a.len() * i + n - 1) / n), int((a.len() * (i + 1) + n - 1) / n)).map(c => c),
            ..range(0, if i == n - 1 { 1 } else { 0 } * (int((a.len() + n - 1) / n) - int(a.len() / n))).map(i => []),
            [#v], ..b.slice(int((a.len() * i + n - 1) / n), int((a.len() * (i + 1) + n - 1) / n)).map(c => c),
            ..range(0, if i == n - 1 { 1 } else { 0 } * (int((a.len() + n - 1) / n) - int(a.len() / n))).map(i => []),
        )
    ).join()
)

#align(center, text(14pt)[
  *Unicode 字符编码全解*
])

#image("pic/charset.png")

想必你也被琳瑯满目的字符编码弄得晕头转向，动不动爆乱码出来骚扰你，烦死了！

UTF-32、UTF-16、UTF-8、ASCII、UCS-4、UCS-2、Latin-1、ISO-8859-1、ANSI、GBK、GB2312、GB18030、BIG5、EUC-KR、Shift-JIS、EUC-JP、EUC-TW、ISO-2022-JP、HZ-GB-2312、UTF-7……

这里面实际上只需要 UTF-32 和 UTF-8 存在就够了。

其余全是历史遗留垃圾！给*历史失误*擦屁股的。

所有的乱码现象，归根结底，是因为 UTF-32 和 UTF-8 无法普及导致的。这些过时字符编码格式的存在，为的就只是给老软件擦屁股，没有任何不可取代的理由。

其实 UTF-8 也是一个历史失误擦屁股的产物，如果能回到过去，我一开始就会责令他们全用 UTF-32，世界再也没有乱码问题！

很多教程的讲解顺序是：ASCII → Latin-1 → GB2312 → GBK → UCS-2 → UTF-16 → UTF-8 → UTF-32

这确实是他们在*历史上发明的顺序*，但*最佳的学习顺序*恰恰应该是反过来。

正如学 C++ 要从 C++23 学起，学字符编码也要从 UTF-32 学起。见过“光明”，再来“倒序”回过去，学习如何应付历史遗留问题，才是好的学习顺序。

= 计算机如何表示文字

人类发明了文字，文字由一个个字符组成。

可计算机只支持 0 和 1，如何表示人类的文字？

如果每个字都当作“点阵图片”存的话，就浪费太多空间了，而且精度不高。

```
0000001000000
0000001000000
0000001000000
0000010100000
0000010100000
0000100010000
0001000001000
0010000000100
0100000000010
1000000000001
```

== 字符集

更高效的方法是，把地球上所有存在的文字罗列出来，排在一个超大的表里，使每个字符都有一个整数作为序号对应，这个一一对应的关系就是字符集。

//(lambda s: print('#codetab("字符", "编号", ("' + '", "'.join(c for c in s) + '"), (' + ', '.join(f'"{ord(c)}"' for c in s) + '), 2)'))(list(map(chr, range(ord('我'), ord('我') + 10))))
#codetab("字符", "编号", ("我", "戒", "戓", "戔", "戕", "或", "戗", "战", "戙", "戚"), ("25105", "25106", "25107", "25108", "25109", "25110", "25111", "25112", "25113", "25114"), 2)

/ 字符集: 从“字符”到“整数”的一一映射。

而这些产生的整数，就被称为“码点 (code point)”。

/ 码点: 字符集中每个字符对应出来的整数。

Unicode 就是目前最好的字符集，他收录了从英文、中文、数字、标点、拉丁文、希腊字母、日文、韩文、梵文、藏文、甲骨文、埃及象形文字、数学符号、特殊符号、Emoji 等等，所有你能想到的的字符。

其他的字符集基本只收录了特定的语言。例如 GB2312 字符集只为中文常用字、部分数学符号、希腊字母、俄文字母、日文片假名等指定了编号，并没有收录韩文、Emoji、埃及象形文字等。

后来 GBK 加入了更多中文的生僻字，解决部分人名打不出的尴尬问题。在 GB18030 字符集中又收录了更多文字和 Emoji 等。

=== 建议用十六进制

//(lambda s: print('#codetab("字符", "编号", ("' + '", "'.join(c for c in s) + '"), (' + ', '.join(f'"{ord(c)}"' for c in s) + '), 2)'))(list(map(chr, range(ord('我'), ord('我') + 10))))
#codetab("字符", "编号", ("我", "戒", "戓", "戔", "戕", "或", "戗", "战", "戙", "戚"), ("25105", "25106", "25107", "25108", "25109", "25110", "25111", "25112", "25113", "25114"), 2)

由于计算机内部都是二进制，而二进制写起来又太长了，程序员一般喜欢用十六进制数表示数字。这样很多东西都会简化，比如 32 位无符号整数类型 (uint32) 能表示数值上限是 4294967295，一个在十进制看来完全是乱码的数字，用十六进制写就是 0xFFFFFFFF，一目了然。

//(lambda s: print('#codetab("字符", "编号", ("' + '", "'.join(c for c in s) + '"), (' + ', '.join(f'"0x{ord(c):X}"' for c in s) + '), 2)'))(list(map(chr, range(ord('我'), ord('我') + 10))))
#codetab("字符", "编号", ("我", "戒", "戓", "戔", "戕", "或", "戗", "战", "戙", "戚"), ("0x6211", "0x6212", "0x6213", "0x6214", "0x6215", "0x6216", "0x6217", "0x6218", "0x6219", "0x621A"), 2)

“我”这个字，在这个表中，编号为 0x6211。于是当计算机需要表示“我”这个字符时，就用 0x6211 这个整数代替。

如果要表示多个字符，那就用一个整数的数组吧！

例如当计算机要处理“我是小彭老师”这段文字，就可以用：

```
0x6211 0x662F 0x5C0F 0x5F6D 0x8001 0x5E08
```

这一串数字代替。

//(lambda s: print('#codetab("字符", "编号", ("' + '", "'.join(c for c in s) + '"), (' + ', '.join(f'"0x{ord(c):X}"' for c in s) + '), 2)'))("我是小彭老师")
#codetab("字符", "编号", ("我", "是", "小", "彭", "老", "师"), ("0x6211", "0x662F", "0x5C0F", "0x5F6D", "0x8001", "0x5E08"), 1)

=== 乱码的本质

随着互联网的发展，由于各国各自发展出了自己的字符集，导致各种编码混乱，国际上的文件交换难以互通，乃至于有些文字在某些国家是看不到的。这种情况下，Unicode 出现了，他的使命就是统一全世界的字符集，保证全世界所有的文字都能在全世界所有的计算机上显示出来。

例如你在玩一些日本的 galgame 时，会发现里面文字全部乱码。这是因为 Windows 在各个地区发行的是“特供版”：在中国大陆地区，他发行的 Windows 采用 GBK 字符集，在日本地区，他发行的 Windows 采用 Shift-JIS 字符集。日本程序员编译程序时，程序内部存储的是 Shift-JIS 的那些“整数”。这导致日本的 galgame 在中国大陆特供的 Windows 中，把 Shift-JIS 的“整数”用 GBK 的表来解读了，从而乱码（GBK 里的日文区域并没有和 Shift-JIS 重叠）。需要用 Locale Emulator 把 Shit-JIS 翻译成 Unicode 读给 Windows 听。如果日本程序员从一开始就统一用 Unicode 来存储，中国区玩家的 Windows 也统一用 Unicode 解析，就没有这个问题。

#fun[奇妙的是，他们美国大区自己的 Windows 系统却是最通用的 UTF-8，偏偏发往中国的特供版 Windows 给你用 GBK……]

乱码的本质就是不同的字符集中，同一个字符对应的整数不同。例如 GBK 中“我”是 0xCED2，而 Unicode 中 0xCED2 就变成 “컒” 了。保存文件的人使用 GBK，打开文件的人使用 Unicode，就会出现“我”变成“컒”的问题，就是所谓的乱码现象了。

导致乱码的罪魁祸首，不仅仅是因为字符集互不兼容，还有一个东西——字符编码，又进一步助纣为虐，增加更多混乱，我们稍后会讲到。

总之，继续用这些不统一的、各自为政的、残缺不全的字符集，没有任何好处，他们存在只是为了让老程序还能继续运行，新项目应当统一采用 Unicode 字符集。

== 字符编码

仅仅是变成了整数还不够，还需要想办法把这些整数进一步翻译成计算机能识别的“字节”，也就是俗称的序列化。

整数翻译到字节的方案有很多种，方案就称为字符编码。

整数叫码点 (code point)，翻译出来的字节叫码位 (code unit)。

#image("pic/charset.png")

/ 字符集: 从字符到整数的一一映射。
/ 码点: 字符集中每个字符对应出来的整数。
/ 字符编码: 把整数进一步映射成一个或多个字节。
/ 码位: 字符编码产生的字节，一个整数可能产生多个字节。

=== 最简单的 UCS-4

Unicode 字符集映射的整数范围是从 0x0 到 0x10FFFF。

这个范围最多只需要 21 二进制位就能存储了。而 C 语言并没有提供 21 位整数类型，他支持的整数类型如下：

#codetab("类型", "最大值", ([`uint8_t`], [`uint16_t`], [`uint32_t`]), ("0xFF", "0xFFFF", "0xFFFFFFFF"), 1)

哪个最适合存储 Unicode 字符呢？似乎只能用 `uint32_t` 了，虽然只利用了 32 位中的最低 21 位，有点浪费。

例如计算机要存储“我是小彭老师!😉”这段文字，首先通过查表：

//(lambda s: print('#codetab("字符", "编号", ("' + '", "'.join(c for c in s) + '"), (' + ', '.join(f'"0x{ord(c):X}"' for c in s) + '), 2)'))("我是小彭老师!😉")
#codetab("字符", "编号", ("我", "是", "小", "彭", "老", "师", "!", "😉"), ("0x6211", "0x662F", "0x5C0F", "0x5F6D", "0x8001", "0x5E08", "0x21", "0x1F609"), 2)

查到每个字对应的整数后，用一个 `uint32_t` 数组存储：

```cpp
vector<uint32_t> str = {0x6211, 0x662F, 0x5C0F, 0x5F6D, 0x8001, 0x5E08, 0x21, 0x1F609};
```

用 32 位整数（即 4 字节）直接表示文本，这就是 UCS-4 编码。

在 UCS-4 中，1 个字符固定对应 4 个字节，非常方便计算机处理。

UCS-4 还有一个别名叫 UTF-32，他们是同一个东西。

=== 阉割的 UCS-2

可实际上 Unicode 中的大多数常用字符都在 0x0 到 0xFFFF 的范围内，超过 0x10000 的基本都是一些 Emoji、特殊符号、生僻字、古代文字，平时很少用到。

#tip[例如 “𰻞𰻞面” 中 “𰻞” 的编号为 0x30EDE，超过了 0xFFFF，但是你平时很少用到。这是 Unicode 委员会有意设计的，他们把越常用的字符越是放在前面。]

例如中文字符的范围是 0x4E00 到 0x9FFF，日文假名的范围是 0x3040 到 0x30FF，韩文字符的范围是 0xAC00 到 0xD7A3，拉丁文字符的范围是 0x0000 到 0x00FF，等等。

#tip[0x4E00 对应的中文字符是“一”，0x9FFF 对应的中文字符是“鿿”，可见 Unicode 委员会对中文区也采用了按笔画多少排序的策略。]

UCS-2：要不然，我们索性放弃支持生僻字符吧！只保留 0x0 到 0xFFFF 范围的常见字符，那 `uint16_t` 就够用了。这样每个字符只占用 2 字节，节省了一半的空间！

用 16 位整数（即 2 字节）表示 0xFFFF 以内字符组成的文本，这就是 UCS-2 字符编码。

因为最大只能表示 16 位整数，超过 0xFFFF 范围的 “𰻞” 字和 “😉” 这种趣味表情符号，就无法显示了。

- UCS-4：“我是小彭老师!我想吃𰻞𰻞面😉”
- UCS-2：“我是小彭老师!我想吃面”

#detail[不过，UTF-16 和 UCS-2 却是不同的，后面讲。]

=== 欧美的 Latin-1

可是，也有一些欧美用户认为，0xFFFF 也太多了！他们平时只需要用英文字母和拉丁字母就够了，几乎用不到中文字符。

好巧不巧，英文和拉丁字母的整数范围是 0x0 到 0xFF。一个 `uint8_t` 就能存储，内存消耗进一步降低到 1 字节。

#image("pic/latin1.svg")

0x0 到 0xFF 内所有的拉丁字母如上图所示。

欧美用户很满意，可是这样就无法支持中文和表情符号了😭

- UCS-4：“我是小彭老师!我想吃𰻞𰻞面😉”
- UCS-2：“我是小彭老师!我想吃面”
- Latin-1：“!”

#fun[欧美用户只需要 `uint8_t` 就行了，而小彭老师要考虑的就多了。]

=== 美国的 ASCII

#fun[¿还有高手?]

美国用户：什么拉不拉丁的，我们只需要英文字母就够了！

好巧不巧，英文字母的整数范围是 0x0 到 0x7F，只需要 7 位。

好处是，无符号 8 位整数 `int8_t` 也能表示了！其中符号位完全不用，剩下 7 位刚刚好。

#image("pic/ascii.png")

如图所示，ASCII 范围中并不是只有英文字母，其中还有数字、标点符号、空格、换行、各种特殊控制字符等。

大部分控制字符都在 0x0 到 0x1F 范围内，只有 DEL 这一个控制字符鹤立鸡群，位于 0x7F。

=== 总结

表示范围上一个比一个小：UCS-4 > UCS-2 > Latin-1 > ASCII

#table(
    columns: 3,
    [字符编码], [能表示的范围], [类型],
    [UCS-4], [0x10FFFF (完整)], [`uint32_t`],
    [UCS-2], [0xFFFF (常用字)], [`uint16_t`],
    [Latin-1], [0xFF (拉丁)], [`uint8_t`],
    [ASCII], [0x7F (不带注音的)], [`int8_t`],
)

这里面除 UCS-4 外，都是不完整的，现代软件理应不再使用他们。

== 变长编码

UCS-2 为了避免浪费，想方设法用 2 字节表示字符，只好放弃超过 0x10000 的生僻字符，只支持了 0x0 到 0xFFFF 范围。

有没有办法既节约到 2 字节空间，又能把全部 0x10FFFF 的字符都容纳下来？UTF-16 应运而生。

想当年 Unicode 委员会确立字符表时，好心把 0xD800 到 0xDFFF 之间这一段区域预留了“空号”。

#image("pic/ucs2range.png")

UTF-16 就是利用了这一段只有 0x800 大小的空号区间，容纳了剩余的 0x100000 个字符。

#fun[怎么可能？]

用一个 0x800 当然装不下，用两个就够了！方案如下：

+ 对于小于 0x10000 的正常字符，和 UCS-2 一样，直接存入 `uint16_t` 数组。
+ 对于 0x10000 到 0x10FFFF 范围的稀有字符：
 + 先把这个整数减去 0x10000，0x10000 会变成 0x0，0x10FFFF 会变成 0xFFFFF。
 + 把这个 0x0 到 0xFFFFF 的 20 位整数按二进制切成两半，每半各 10 位。
 + 把低半段加上 0xD800，高半段加上 0xDC00。
 + 得到两个 0xD800 到 0xDFFF 之间的数，正好 0xD800 到 0xDBFF 范围内的数用来表示前半段，0xDC00 到 0xDFFF 范围内的数用来表示后半段。
 + 把这两个数依次存入 `uint16_t` 数组。

所有的稀有字符，都会被拆成两段：

- 0xD800 到 0xDBFF 范围内的数用来表示一个大数的前半段，称为低代理对 (low-surrogates)。
- 0xDC00 到 0xDFFF 范围内的数用来表示一个大数的后半段，称为高代理对 (high-surrogates)。

例如生僻字“𰻞”的编号为 0x30EDE，减去 0x10000 得到 0x20EDE，变成二进制就是 0b10000011101101110。

把这 20 位二进制拆成两半，低半段是 0b10000，加上 0xD800 得到 0xD840，高半段是 0b0111011010，加上 0xDC00 得到 0xDC6A。

所以生僻字“𰻞”的 UTF-16 编码后，塞入 `uint16_t` 数组是这样：
```cpp
std::vector<uint16_t> biang = {0xD840, 0xDC6A};
```

所以，UTF-16 是一个变长编码格式，一个字符可能需要两个数才能表示。

#fun[这就是为什么说 UTF-16 和 UCS-2 是不同的。UCS-2 是定长编码，不能表示完整的 0x10FFFF 范围，而 UTF-16 可以，代价就是他成了变长编码，需要做额外的特殊压缩处理。]

变长编码有许多严重的问题，

== C 语言的问题

=== C 语言之殇 `char`

这就是为什么 C 语言的 `char` 类型是 8 位有符号整数类型 `int8_t`。

毕竟设计之初，C 语言的 `char` 就只考虑了 ASCII 的字符，没想过支持 UCS-4。

=== `wchar_t` 拯救世界

C 语言为了支持 UCS-4，就引入了 32 位无符号整数类型 `wchar_t`。

`wchar_t` 是一个编译器内置类型，和 `char` 一样是语言的一部分。

而 `int8_t` 只是 `stdint.h` 头文件中 `typedef` 定义的类型别名，不导入那个头文件就不会有。

```cpp
// stdint.h
typedef signed char uint8_t;
typedef signed short uint16_t;
typedef signed int uint32_t;
typedef signed long long uint64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
```

=== 真的有这么简单的事吗？

`wchar_t` 在 Linux 上是 32 位的（理应如此）。

可 `wchar_t` 在 Windows 上却是 16 位的（由于讨厌的历史原因）！

== GB2312

可当时美国程序员偷懒，哪想到后来会加入那么多字符。

他们发明了 ASCII 字符集，给每个英文字母、数字和标点符号赋予了 0x0 到 0x7F 的整数。

于是 C 语言就用 1 字节的 char 来表示 ASCII 字符，范围也只能是 0x0 到 0x7F。

后来计算机引入欧洲其他国家，他们的语言需要使用一些带注音符号的字母，他们发现 ASCII 只占据了 char 的 0x0 到 0x7F 范围，于是把 0x80 到 0xFF 范围的整数，用来映射自己的各种带注音字母。就有了 Latin-1，由于 C 语言的 char 本来就能表示 8 位整数，所以过去写的只支持 ASCII 的程序直接就能用于处理 Latin-1，无缝衔接。

后来计算机更加普及，引入中国时，发现有几千多个汉字需要表示，0x80 到 0xFF 才 128 个空位，根本塞不下。他们想过把 char 改成 16 位的，但是 C 语言标准已定，Windows 是闭源软件又无法修改，为了兼容已经适配 ASCII 的 Windows 和各种 C 语言软件，中国的计算机学家们只好想出一个歪招：

=== UCS-2

最终在 Windows NT 这个版本中，
