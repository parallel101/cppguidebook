// Load CJK and monospace fonts from CDN
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
