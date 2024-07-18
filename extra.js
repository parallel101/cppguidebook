var footer = document.querySelector('footer');
footer.innerHTML = footer.innerHTML.replace('<p>Documentation built with <a href="https://www.mkdocs.org/">MkDocs</a>.</p>', '<p style="display: none">本页面由 <a href="https://www.mkdocs.org/">MkDocs</a> 构建</p>');

var lut = {
    Next: '下一章',
    Previous: '上一章',
    Search: '搜索',
    'Edit on GitHub': '编辑此页面',
};

// enumerate all nav-links
var links = document.querySelectorAll('a.nav-link');
for (var i = 0; i < links.length; i++) {
    var link = links[i];
    for (var j = 0; j < link.childNodes.length; j++) {
        var node = link.childNodes[j];
        if (node.data !== undefined) {
            var key = node.data.trim();
            var to = lut[key];
            if (to !== undefined) {
                node.data = node.data.replace(key, to);
            } else {
                console.log({a: node.data});
            }
        }
    }
}
