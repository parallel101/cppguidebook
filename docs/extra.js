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
            }
        }
    }
}

var nasms = document.querySelectorAll('code.language-nasm');
for (var i = 0; i < nasms.length; i++) {
    var nasm = nasms[i];
    // add class .language-wasm .hljs
    nasm.classList.add('language-x86asm', 'hljs');
    // remove class .language-nasm
    nasm.classList.remove('language-nasm');
}

var stylesheets = [
    'https://cdn.jsdelivr.net/npm/@fontsource/noto-sans-sc@5.0.19/index.min.css',
    'https://cdn.jsdelivr.net/npm/jetbrains-mono@1.0.6/css/jetbrains-mono.min.css',
    'https://cdn.jsdelivr.net/npm/@fontsource/noto-serif-sc@5.0.13/chinese-simplified-500.min.css',
];
for (var i = 0; i < stylesheets.length; i++) {
    var link = document.createElement('link');
    link.setAttribute('rel', 'stylesheet');
    link.setAttribute('href', stylesheets[i]);
    document.head.appendChild(link);
}
