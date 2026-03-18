// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="index.html"><strong aria-hidden="true">1.</strong> 小彭老师现代 C++ 大典</a></li><li class="chapter-item expanded "><a href="hello_world.html"><strong aria-hidden="true">2.</strong> 你好，世界</a></li><li class="chapter-item expanded "><a href="platform.html"><strong aria-hidden="true">3.</strong> 开发环境与平台选择</a></li><li class="chapter-item expanded "><a href="variable_types.html"><strong aria-hidden="true">4.</strong> 变量与类型</a></li><li class="chapter-item expanded "><a href="functions.html"><strong aria-hidden="true">5.</strong> 认识函数</a></li><li class="chapter-item expanded "><a href="auto.html"><strong aria-hidden="true">6.</strong> auto 神教</a></li><li class="chapter-item expanded "><a href="symbols.html"><strong aria-hidden="true">7.</strong> 重新认识声明与定义</a></li><li class="chapter-item expanded "><a href="struct_layout.html"><strong aria-hidden="true">8.</strong> 结构体布局</a></li><li class="chapter-item expanded "><a href="cpp_tricks.html"><strong aria-hidden="true">9.</strong> 应知应会 C++ 小技巧</a></li><li class="chapter-item expanded "><a href="lambda.html"><strong aria-hidden="true">10.</strong> 小彭老师带你学函数式编程</a></li><li class="chapter-item expanded "><a href="type_rich_api.html"><strong aria-hidden="true">11.</strong> 现代化的 API 设计指南</a></li><li class="chapter-item expanded "><a href="no_more_new.html"><strong aria-hidden="true">12.</strong> 现代 C++ 从拒绝 new 开始</a></li><li class="chapter-item expanded "><a href="stl_map.html"><strong aria-hidden="true">13.</strong> STL 精讲：std::map 和他的朋友们</a></li><li class="chapter-item expanded "><a href="design_overview.html"><strong aria-hidden="true">14.</strong> 设计模式总览</a></li><li class="chapter-item expanded "><a href="design_virtual.html"><strong aria-hidden="true">15.</strong> 让虚函数再次伟大！</a></li><li class="chapter-item expanded "><a href="design_gamedev.html"><strong aria-hidden="true">16.</strong> 游戏开发中常用的设计模式</a></li><li class="chapter-item expanded "><a href="design_functor.html"><strong aria-hidden="true">17.</strong> 函数式设计模式</a></li><li class="chapter-item expanded "><a href="design_variant.html"><strong aria-hidden="true">18.</strong> 静态多态与面向数据编程</a></li><li class="chapter-item expanded "><a href="design_erasure.html"><strong aria-hidden="true">19.</strong> 类型擦除神教及其实现</a></li><li class="chapter-item expanded "><a href="design_concept.html"><strong aria-hidden="true">20.</strong> 鸭子类型与 C++20 concept</a></li><li class="chapter-item expanded "><a href="error_code.html"><strong aria-hidden="true">21.</strong> 现代 C++ 错误处理知多少</a></li><li class="chapter-item expanded "><a href="cpp_lifetime.html"><strong aria-hidden="true">22.</strong> 深入理解析构函数与生命周期</a></li><li class="chapter-item expanded "><a href="cpp_memory.html"><strong aria-hidden="true">23.</strong> 真正的 C++ 内存模型！</a></li><li class="chapter-item expanded "><a href="unicode.html"><strong aria-hidden="true">24.</strong> 字符编码那些事</a></li><li class="chapter-item expanded "><a href="threading.html"><strong aria-hidden="true">25.</strong> C++ 多线程编程</a></li><li class="chapter-item expanded "><a href="test_and_safe.html"><strong aria-hidden="true">26.</strong> 测试与安全话题</a></li><li class="chapter-item expanded "><a href="undef.html"><strong aria-hidden="true">27.</strong> 未定义行为完整列表</a></li><li class="chapter-item expanded "><a href="llvm_intro.html"><strong aria-hidden="true">28.</strong> 小彭老师带你学 LLVM</a></li><li class="chapter-item expanded "><a href="cuda_intro.html"><strong aria-hidden="true">29.</strong> 现代 C++ 的 CUDA 编程</a></li><li class="chapter-item expanded "><a href="cmake_tutor.html"><strong aria-hidden="true">30.</strong> 学现代 C++ 从现代 CMake 学起</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded "><a href="about.html"><strong aria-hidden="true">31.</strong> 关于小彭老师</a></li><li class="chapter-item expanded "><a href="donate.html"><strong aria-hidden="true">32.</strong> 赞助名单</a></li><li class="chapter-item expanded "><a href="interview.html"><strong aria-hidden="true">33.</strong> 小彭老师面试经验</a></li><li class="chapter-item expanded "><a href="recommend.html"><strong aria-hidden="true">34.</strong> 参考资料与项目推荐</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
