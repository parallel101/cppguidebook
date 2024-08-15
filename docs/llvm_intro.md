# 小彭老师带你学 LLVM

[TOC]

## LLVM 介绍

LLVM 是一个跨平台的编译器基础设施，它不是一个单一的编译器，而是一系列工具和库的集合，其提供丰富的数据结构 (ADT) 和中间表示层 (IR)，是实现编译器的最佳框架。

LLVM 是编译器的中后端，中端负责优化，后端负责最终汇编代码的生成，他并不在乎调用他的什么高级语言，只负责把抽象的代数运算，控制流，基本块，转化为计算机硬件可以直接执行的机器码。

Clang 只是 LLVM 项目中的一个前端，其负责编译 C/C++ 这类语言，还有用于编译 Fotran 的 Flang 前端。除此之外，诸如 Rust、Swift 之类的语言，也都在使用 LLVM 做后端。

> {{ icon.story }} 举个例子，析构函数在 `}` 处调用，这是 C++ 的语法规则，在 Clang 前端中处理。当 Clang 完成 C++ 语法规则，语义规则的解析后，就会调用 LLVM，创建一种叫中间表示码（IR）的东西，IR 介于高级语言和汇编语言之间。IR 是为了统一来自不同语言，去往不同的一层抽象层。一是便于前端的统一实现，Clang 这样的前端只需要生成抽象的数学运算，控制流这些 IR 预先定义好的指令就可以了，不用去专门为每个硬件设计一套生成汇编的引擎；二是 LLVM IR 采用了对优化更友好的 SSA 格式，而不是糟糕的寄存器格式，大大方便了优化，等送到后端的末尾时才会开始将 IR 翻译为汇编代码，最终变成可执行的机器码。

如果没有 IR 会怎样？假设有 $M$ 种语言，$N$ 种硬件，就需要重复实现 $M \times N$ 个编译器！而 IR 作为中间表示层，令语言和硬件的具体细节解耦了，从而只需要写 $M + N$ 份代码就可以：语言的开发者只需要考虑语法如何变成数学运算和控制流，硬件厂商只需要考虑如何把数学和跳转指令变成自己特定的机器码。因此，不论是 LLVM/Clang 还是 GCC 家族，跨平台编译器内部都无一例外采用了 IR 做中间表示。

> {{ icon.story }} 有了统一的抽象 IR 以后，不管你是 C++ 析构函数还是 C 语言普通函数，进了 IR 以后都是一样的函数调用，减轻了编译器中后端开发者的心智负担。要开发一种新语言，只管解析完语法生成 IR 输入 LLVM，他会替你包办好优化和汇编的事。

### 参考资料

- LLVM 官方仓库：https://github.com/llvm/llvm-project
- LLVM 用户文档：https://llvm.org/docs/
- LLVM 源码级文档：https://llvm.org/doxygen/
- 《Learn LLVM 17》：https://github.com/xiaoweiChen/Learn-LLVM-17

## 为什么选择 LLVM

- 如果你对 C++ 语言的底层实现感兴趣，编译器是绕不过的一环。可御三家中，MSVC 是闭源的无法学习，GCC 代码高度耦合，且很多原始的 C 语言“古神低语”混杂其中，可读性较差。Clang 是一个跨平台的 C++ 编译器前端，而 LLVM 正是他的后端，高度模块化的设计，代码质量优秀，且很容易加入自己的新模块，最适合编译器新人上手学习。而除去 Clang 负责的 C++ 语法解析后，LLVM 后端占据了半壁江山。你想不想探究编译器是如何利用未定义行为优化的？想不想知道为什么有时 C++ 编译器出现异常的行为？想不想了解怎样才能写出对编译器友好的代码，方便 Clang 和 LLVM 自动帮你优化？那就来学习 LLVM 吧！
- 前端和后端众多，无论你是打算开发一种新型语言，还是自研一种新的 CPU 架构，考虑支持 LLVM 作为中端几乎是你唯一的选择。
- 对于 CPU/GPU 硬件厂商而言：由于丰富的前端，支持 LLVM 将使你的硬件直接支持 C/C++/CUDA/OpenCL/SyCL/Objective-C/Fortran/Rust/Swift 等所有 LLVM 有前端的语言。例如有的国产显卡基于 LLVM 添加了自己的硬件指令集作为后端，然后再利用 LLVM 的 CUDA 前端，就实现了兼容 CUDA，AMD 得以实现 CUDA 兼容也是基于此。反之，新语言也可以使用 LLVM 的 PTX 后端输出，从而支持在 NVIDIA 显卡上执行。
- 对于想发明新语言或为现有脚本语言实现 JIT 加速的开发者而言：由于丰富的后端，新语言使用 LLVM 就能直接支持 x86/ARM/MIPS/PPC/BPF/PTX/AMDGPU/SPIR-V 等各种架构和指令集，而自己不用增加任何底层细节负担。例如 Rust 虽然宣称可以取代 C++，但最终仍是调用 LLVM 实现编译，产生可以执行的二进制码，自己一个个适配所有硬件平台的成本实在太高了，且不论还要专门开发所有的优化 pass，而 LLVM 作为业界支持最完善的现成品在很长一段时间内都很难代替。
- 中端优化和分析能力强大，新语言若基于 LLVM，优化方面的工作都有现成的实现，可以全部让 LLVM 代劳，自己只需要负责解析语法，生成 LLVM IR 即可，如何优化后生成二进制码根本无需操心，LLVM 会自动根据当前的目标平台判断。
- 高度自包含，完全基于 CMake 的模块化构建，充满现代感。用户可自行选择要构建的模块。且几乎完全无依赖就能构建，有 CMake 有编译器就行，无需安装繁琐的第三方库。相比之下 GCC 采用落后的 Makefile + AutoConf 构建系统，且版本要求苛刻。
- LLVM 采用的 MIT 开源协议十分宽松，对商用自由度较高。且代码质量优秀，容易自己插入新功能，可修改后供自己使用，因此常用于闭源驱动中（例如 NVIDIA 的 OpenGL 驱动等）。相比之下 GCC 采用的 GPL 协议就比较严格，不得自己修改后闭源发布（必须连同源代码一起发布）。
- LLVM 附带了许多实用命令行工具，帮助我们分析编译全过程的中间结果，理解优化是如何发生的。例如 llvm-as（LLVM IR 转为压缩的字节码），llvm-dis（字节码转为 IR），opt（可以对 IR 调用单个优化 pass），llc（将字节码转换为目标机器的汇编代码），llvm-link（IR 级别的链接，输入多个字节码文件，产生单个字节码文件），lld（对象级别的链接，类似于 GNU ld），lli（解释执行字节码），llvm-lit（单元测试工具）。
- 一些芯片相关的大厂中，编译器方面的岗位需求量很大。而其中主要用的，例如 NVIDIA 的编译器 nvcc，其后端就是基于 LLVM 魔改的，因此学习 LLVM 很有就业前景。

> {{ icon.detail }} 为什么有了 Clang 还要 nvcc？虽然 Clang 也能支持 CUDA，但 Clang 只能把 CUDA 编译成所有 NVIDIA 显卡都能通用的 PTX，无法生成专门对不同显卡型号特化 SASS 汇编（需要调用 NVIDIA CUDA Toolkit 提供的 ptxas 才能转换）。而 nvcc 的前端除了是自己的，后端同样是调用 LLVM 生成 PTX 汇编，只是 NVIDIA 对 LLVM 做了一些闭源的魔改（其实早期 nvcc 的后端是基于 NVIDIA 自研的 NVVM 后端，但是发现效果不好，最近正在逐步切换到 LLVM 后端，毕竟是老牌项目）。如果对 C++ 新特性有追求，可以用 Clang 前端 + LLVM 生成 PTX + ptxas 汇编的组合，实现自由世界的 CUDA 工作流（之后介绍）。但是因为 ptxas，以及 CUDA 其他运行时库的需要，Clang CUDA 依然需要安装 CUDA Toolkit 才能正常运行，且对 CUDA 版本要求比较严格，可能需要较多的配置功夫。

### LLVM 上下游全家桶的宏伟图景

LLVM 项目不仅包含了 LLVM 本体，还有一系列围绕 LLVM 开发的上下游工具。例如 Clang 编译器就是 LLVM 项目中的一个子项目，他是一个 C/C++/CUDA/OpenCL/SyCL/Objective-C 等 C 类语言的前端，只负责完成语法的解析，实际编译和二进制生成交给 LLVM 本体（中后端）来处理。通常说的 LLVM 指的是 LLVM 本体，其是一个通用的编译器基建，仅包含中端（各种优化）和后端（生成 x86/ARM/MIPS 等硬件的指令码）。Clang 解析 .cpp 文件后产生 IR，调用 LLVM 编译生成的 .o 对象文件，又会被输入到同属 LLVM 项目的一个子项目：LLD 链接器中，链接得到最终的单个可执行文件（.exe）或动态链接库（.dll），LLD 还可以开启链接时优化，这又会用到 BOLT 这个链接时优化器，对生成的单个二进制做进一步汇编级别的优化。不仅如此，著名的 C++ 标准库实现之一，libc++，也是 LLVM 项目的一部分，相比 GCC 家族的 libstdc++ 更简单，更适合学习。不仅如此，还有并行的 STL 实现 pstl，OpenCL 编译器 libclc 等……应有尽有，是编译器开发者的天堂。

Clang 编译 C++ 程序的整个过程：

**Clang 前端解析 C++ 语法 -> LLVM 中端优化 -> LLVM 后端生成指令码 -> LLD 链接 -> BOLT 链接后优化**

而 GCC 就没有这么模块化了，虽然 GCC 内部同样是有前端和中端 IR，但是整个就是糊在一个 GCC 可执行文件里，难以重构，积重难反，也难以跨平台（MinGW 还是民间自己移植过去的，并非 GCC 官方项目）。和 Clang 能轻易作为 libclang 和 libLLVM 库发布相比，高下立判。MSVC 更是不必多说，连源码都开放，让人怎么学习和魔改啊？

### 学习 LLVM 前的准备

要学习 LLVM，肯定不能纸上谈兵。LLVM 是开源软件，最好是自己下载一个 LLVM 全家桶源码，然后自己从源码构建。

注意：我们最好是从源码构建 LLVM 和 Clang，方便我们动手修改其源码，添加模块，查看效果。下载二进制发布版 LLVM 或 Clang 的话，虽然同样可以使用所有的命令行工具，就只能对着 IR 一通分析盲猜了。

虽然 LLVM 几乎是无依赖的，只需要 CMake 和编译器就能构建，但依然推荐使用 Linux 系统进行实验，以获得和小彭老师同样的开发体验。Windows 用户建议使用 Virtual Studio 或 CLion 等强大 IDE 帮助阅读理解源码；Linux 用户建议安装 [小彭老师 vimrc](https://github.com/archibate/vimrc)；或者如果你是远程 Linux，可以试试看 VSCode 的远程 SSH 连接插件；CLion 似乎也有远程插件，只不过需要在远程安装好客户端。

> {{ icon.tip }} 强大的 IDE 和编辑器对学习任何大型项目都是必不可少的，特别是跳转到定义，以及返回这两个操作，是使用频率最高的，在源码之间的快速跳转将大大有助于快速。

> {{ icon.tip }} 如果实在没有条件自己构建 LLVM 源码，或者 IDE 比较拉胯：可以去 LLVM 的在线源码级文档（使用 Doxygen 生成）看看。其不仅提供了 LLVM 中所有类和函数的详尽文档，参数类型，用法说明等；还提供了每个函数的所在文件和行号信息，点击类型或函数名的超链接，就可以在源码和文档之间来回跳转。还能看到哪里引用了这个函数，还能显示类的继承关系图，非常适合上班路上没法打开电脑时偷学 LLVM 源码用。例如，`llvm::VectorType` 这个类的文档：https://llvm.org/doxygen/classllvm_1_1VectorType.html

## LLVM 开发环境搭建

### 环境准备

LLVM（和 Clang）的构建依赖项几乎没有，只需要安装了编译器和 CMake 就行，非常的现代。

#### Linux/MacOS 用户

首先安装 Git、CMake、Ninja、GCC（或 Clang）。

> {{ icon.tip }} 其中 Ninja 可以不安装，只是因为 Ninja 构建速度比 Make 快，特别是当文件非常多，而你改动非常少时。而且 Ninja 默认就开启多核并行构建，所以大型项目通常会尽量给 CMake 指定 `-G Ninja` 选项，让其使用更高效的 Ninja 后端构建。

Arch Linux:

```bash
sudo pacman -S git cmake ninja gcc
```

Ubuntu:

```bash
sudo apt-get install git cmake ninja-build g++
```

MacOS:

```bash
brew install git cmake ninja gcc
```

开始克隆项目（需要时间）：

```bash
git clone https://github.com/llvm/llvm-project
```

如果你的 GitHub 网速较慢，可以改用 Gitee 国内镜像（只不过这样你就没法给 LLVM 官方水 PR 了 🤣）：

```bash
git clone https://gitee.com/mirrors/LLVM
```

#### Windows 用户

即使是 LLVM 这样毫无依赖项的项目，“只需要安装了编译器和 CMake 就行”，在 Windows 用户看来依然非常科幻。

好在微软也意识到了自己的残废，现在 Virtual Studio 2022 已经替你包办好了（自带 Git、CMake 和 Ninja 了）。

如果你是用 VS2022 自带的 Git 克隆 llvm-project，记得 cd 到 llvm 文件夹里再用 cmake，然而贵物 IDE 的一个 cd 都是如此的困难。

所以这边建议你直接先把 llvm-project 作为 ZIP 下载下来，然后打开其中的 llvm 子文件夹，然后用 VS2022 打开其中的 CMakeLists.txt，然后开始构建。

然后，要开启一个 CMake 选项 `-DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra"`，才能构建 Clang 子项目（否则构建的是赤膊 LLVM，没有任何前端，这毫无意义）。仅此是指定这一个小小选项对于 IDE 受害者又是何等的困难……他们需要在 VS2022 中打开 CMakeSettings.json，修改 x64-Debug 的配置，点击添加一个变量 LLVM_ENABLE_PROJECTS，值为 "clang;clang-tools-extra"……如果他们要改成 Release 配置，又要点击加号创建 x64-Release（千万别点错成 x86-Release！），然后再次点击添加一个变量 LLVM_ENABLE_PROJECTS……

因为 llvm-project 是许多项目的集合，根目录里并没有 CMakeLists.txt，而 VS2022 似乎只能识别根目录的 CMakeLists.txt……

> {{ icon.fun }} 正常系统只需要给你写一串命令，你只管复制粘贴到 Shell 里一执行就搞定了。脑瘫系统需要大量无谓的文字描述和截图箭头指示半天，还经常有人看不懂，要反复强调，画箭头，加粗字体，才能操控他的鼠标点击到正确按钮上。我也想把鼠标宏录下来，可是不同电脑分辨率不同，窗口位置又很随机，电脑响应速度又随机，有时候 C 盘，有时候又 D 盘，根本不给一个统一的操作方式,统一的命令行就没有这种烦恼。所以，能卸载的卸载，能双系统的双系统，能 WSL 也总比腱鞘粉碎器（鼠标）好，至少能一键粘贴小彭老师同款操作。

### 项目目录结构

```
$ cd llvm-project
$ ls
bolt                CONTRIBUTING.md      LICENSE.TXT  pstl
build               cross-project-tests  lld          pyproject.toml
build.sh            flang                lldb         README.md
clang               libc                 llvm         runtimes
clang-tools-extra   libclc               llvm-libgcc  SECURITY.md
cmake               libcxx               mlir         third-party
CODE_OF_CONDUCT.md  libcxxabi            openmp       utils
compiler-rt         libunwind            polly
```

- 注意到这里面有很多的子项目，其中我们主要学习的就是这里面的 llvm 文件夹，他是 LLVM 的本体。其中不仅包含 LLVM 库，也包含一些处理 LLVM IR 和字节码的实用工具（例如 llvm-as）。
- 其次就是 clang 文件夹，这个子项目就是大名鼎鼎的 Clang 编译器，他也是基于 LLVM 本体实现的，本身只是个前端，并不做优化和后端汇编生成。
- clang-tools-extra 这个子项目是 clangd、clang-tidy、clang-format 等 C/C++ 代码质量工具，可以选择不构建。
- libc 是 Clang 官配的 C 标准库，而 libcxx 是 Clang 官配的 C++ 标准库，想学标准库源码的同学可以看看。
- flang 是 LLVM 的 Fortran 前端，编程界的活化石，没什么好说的。
- lldb 是 LLVM 官方的调试器，对标 GCC 的 gdb 调试器，VSCode 中的调试默认就是基于 lldb 的。
- lld 是 LLVM 官方的二进制链接器，对标 GCC 的 ld 和 ld.gold；而 bolt 是链接后优化器，用的不多。
- compiler-rt 是诸如 AddressSantizer（内存溢出检测工具）、MSAN（内存泄漏检测）、TSAN（线程安全检测）、UBSAN（未定义行为检测）等工具的实现。
- mlir 是 LLVM 对 MLIR 的编译器实现（一种为机器学习定制，允许用户自定义新的 IR 节点，例如矩阵乘法等高阶操作，方便特定硬件识别到并优化成自研硬件专门的矩阵乘法指令，最近似乎在 AI 孝子中很流行）。
- libclc 是 LLVM 对 OpenCL 的实现（OpenCL 语言规范的编译器），OpenCL 是孤儿，没什么好说的。
- openmp 是 LLVM 对 OpenMP 的实现（一种用于傻瓜式 CPU 单机并行的框架，用法形如 `#pragma omp parallel for`）。
- pstl 是 LLVM 对 C++17 Parallel STL 的实现（同样是单机 CPU 并行，优势在于利用了 C++ 语法糖，也比较孤儿，用的不多）。
- cmake 文件夹并不是子项目，而是装着和 LLVM 相关的一些 CMake 脚本文件。
- build 文件夹是使用过 CMake 后会生成的一个文件夹，其中存储着构建的全部中间文件和最终的二进制可执行文件。所有的可执行文件都放在 build/bin 子文件夹中，例如 build/bin/llvm-as。

### 开始构建

```bash
cd llvm-project
bash build.sh
```

`build.sh` 脚本的内容等价于：

```bash
cmake -Sllvm -Bbuild -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -GNinja
ninja -Cbuild
```

> {{ icon.tip }} 你在命令行手动输入这两条命令也是等价的，`build.sh` 只是为了方便。

此处 `-S llvm` 选项表示指定源码路径为根目录下的 `llvm` 子项目文件夹，和 `cd llvm && cmake -B build` 等价，但是不用切换目录。

`-G Ninja` 表示使用 Ninja 后端，如果你没有 Ninja，可以去掉该选项，CMake 将会采用默认的 Makefile 后端（更慢）。

> {{ icon.fun }} 如果你是 Wendous 受害者，请自行用鼠标点击序列在 VS2022 中模拟以上代码之同等效果，祝您腱鞘愉快！

`-DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra"` 表示启用 `clang` 和 `clang-tools-extra` 两个子项目。

这是因为通常用的前端都是 C++，所以 LLVM 官方在 `build.sh` 里就这么写了。

> {{ icon.fun }} 如果你口味比较重，想研究 Fortran 前端，也可以定义该 CMake 变量为 `-DLLVM_ENABLE_PROJECTS="flang"`。

`build.sh` 后，需要花费大约 10 分钟时间（取决于你的电脑配置），这段时间你可以先看下面的基本概念速览。等风扇停了以后，LLVM 和 Clang 就构建好了。

### 运行试试

```bash
ls build/bin
```

## 基本概念速览

学过 C 语言的同学都知道，一个 C/C++ 源码文件到计算机实际可执行的 EXE 文件之间，主要有两步操作：编译（compile）和链接（link）。

![](img/compile-3-stage.png)

之所以把编译和链接分开，是因为一个项目常常由许多源码文件组成，而不只是单个文件。编译器把 C++ 源码编译成中间对象文件（.o 或 .obj 格式），如果有很多 .cpp 文件，就会得到很多 .o 文件，然后由链接器负责统一链接所有 .o 文件，就得到了最终的 .exe 或 .dll 目标文件。

分离多 .cpp 文件的好处是，编译速度更快，可以并行编译。而且修改了其中一个 .cpp 文件，只需要重新编译那个 .cpp 对应的 .o 文件，然后重新链接最终的 .exe 即可，无需再重复编译其他 .cpp 文件的 .o 文件了。自动检测哪些 .cpp 文件更新了，需要重新编译 .o 文件，是 Makefile 和 Ninja 之类构建系统的职责。

我们现在要来学习的就是其中的编译阶段，这也是大部分人关注的重点。

编译器是如何将 .cpp 文件转换为充斥着机器指令码 .o 文件的？

> {{ icon.detail }} .o 文件里几乎全是完成的机器指令码，除了部分 call 到外部函数的一部分指令，会留白，等待链接阶段时，由链接器在其他 .o 文件中找到相同的符号时替换上正确的地址和偏移量。

过去，我们把编译器看作黑箱，进去源码，出来机器码，只能认为是魔法。

现在，有了 LLVM 和 Clang 源码在手，终于可以一探究竟了。

实际上，编译这一过程，还可以进一步拆分成三个阶段。

### 编译器的前、中、后端

编译器（Compiler）的工作流程可以分为三个阶段：

1. 前端（Front-end）：负责接收源代码，解析出抽象语法树（AST），并进行语法和语义分析，生成中间表示码（IR）。
2. 中端（Middle-end）：负责优化中间表示码。
3. 后端（Back-end）：负责将优化完毕的中间表示码翻译成机器码。

> {{ icon.tip }} 链接阶段（Link）属于链接器，不算编译器的工作职责；前、中、后端只是对编译（Compile）这一阶段的进一步拆分。

- 如果你想要研究 C++ 语法规则，比如做个 C++ 语法高亮插件，那就需要看前端。libclang 和 clangd 可以帮助你解析 C++ 繁琐的语法，并以 AST 树的结构提供给你。不仅如此，如果你要设计一门新语言，甚至是 OpenGL 驱动（其需要实现 GLSL 编译器），实际上也就是为 LLVM 添加一个前端。
- 如果你对内存模型，性能优化感兴趣，那就去研究中端。这是目前学术研究比较活跃的领域，特别是多面体优化方向，可以尝试水两张 paper 或 PR。这部分都是基于 LLVM IR 操作的，有特别多的算法和数据结构。
- 如果你对汇编语言，机器指令，硬件架构感兴趣，那就去看后端。这里面有把中间表示码翻译成真正可执行的汇编指令的完整过程，自研芯片的大厂通常想要为 LLVM 添加后端。

接下来，让我们走进 LLVM 这座开源工厂，一步步观察一段 C++ 代码被编译成汇编的全过程。

#### 语法树（AST）

编译器的前端负责解析源代码，生成抽象语法树（Abstract Syntax Tree，AST）。AST 是源代码的一种抽象表示，其中每个节点代表源代码中的一个语法结构，例如 if、while、for、函数调用、运算符、变量声明等。每个 AST 节点都有自己的属性，例如类型、作用域、修饰符等。

这部分在 clang 项目中。

clang 解析源码生成语法树的案例：

```cpp
#include <cstdio>

int main() {
    printf("Hello, world!");
    return 0;
}
```

运行命令：

```bash
clang -fsyntax-only -Xclang -ast-dump test.cpp
```

- `-fsyntax-only` 意味着只解析语法，不进行编译和链接；
- `-Xclang` 是指向 Clang 核心传递一个选项，也就是后面紧挨着的 `-ast-dump`；
- `-ast-dump` 是 Clang 核心的选项，表示要求打印出语法树。

输出：
![](img/clang-ast-example.png)
> {{ icon.tip }} 已省略 `<cstdio>` 头文件部分的语法树节点，仅展示了 main 的部分，否则就太长了。

+ 此处 FunctionDecl 就表明，该节点是一个函数（Function）的声明（Decleration）。注意到后面跟着许多和该函数定义有关的关键信息，让我们逐一分析：
    - 这里的十六进制数 `0x567bdbf246d8` 是 AST 节点在编译器内存中的地址，每次都不一样，无意义。
    - 后面的尖括号 `<a.cpp:3:1>` 里还好心提醒了函数定义的位置。
    - 最后是函数名 `main` 和函数类型 `int ()`。

> {{ icon.story }} 该节点的类型是 FunctionDecl，翻译成中文就是函数声明。但是我们写的明明是一个函数的**定义**啊！为什么被 Clang AST 当作了**声明**呢？原来，C++ 官方的话语中，定义也是声明！但声明不都是定义。所以这里的 FunctionDecl 实际上是一个通用的节点，既可以是声明（后面直接接 `;` 的），也可以是定义（后面接着 `{}` 的），要根据是否有子节点（花括号语句块）来判断。

> {{ icon.detail }} 总之，定义和声明是子集关系。当我们要强调一个声明只是声明，没有定义时，会用**非定义声明**这样严谨的律师说法。但日常提问时你说“声明”我也明白，你指的应该是非定义声明。更多相关概念请看[重新认识声明与定义](symbols.md)章节和[白律师的锐评](https://github.com/parallel101/cppguidebook/pull/23)，用文氏图来画就是：![](img/decl-vs-def.png){width=150px}

+ 函数定义节点又具有一个子节点，类型是 CompoundStmt。这个实际上就是我们所说的花括号语句块 `{}` 了。他本身也是一条语句，但里面由很多条子语句组成。规定函数声明 FunctionDecl 如果是定义，则其唯一子节点必须是语句块类型 CompoundStmt，也就是我们熟悉的函数声明后紧接着花括号，就能定义函数。如果是非定义声明（仅声明，不定义）那就没有这个子节点。

#### 中间表示码（IR）

##### 轶事：LLVM IR 为什么不跨平台

Clang 编译时是什么平台就是什么平台了，不同目标平台的 IR 会有些微的不一样（但 IR 类型都是固定的那几个，除了部分特殊硬件 intrinsics），IR 永远只能变成指定目标平台的机器码。

虽然 IR 是通用的中间表示层，但类型大小，矢量宽度等信息和硬件高度绑定，而且有时用户需要根据 `#ifdef __x86_64__` 判断，针对不同的硬件，使用不同的 intrinsics。而 intrinsics 产生的 IR 节点是和硬件高度相关的，无法在其他平台通用。

总之，因为这样那样的原因，LLVM IR 并不支持跨平台共用，不同平台上 Clang 编译出来的 IR 是不同的。

> {{ icon.story }} 也有一些支持跨平台的 IR，比如 SPIR-V 和 MLIR，适用于游戏客户端部署的场景。但显然 LLVM 作为追求极致优化的裸硬件编译器，其 LLVM IR 如果要求跨平台会很不利于 Clang 前端支持硬件 intrinsics，也不利于 LLVM 中端针对目标硬件特性做优化，也会无法支持内联汇编，所以就放弃了。所以现实中，人们会先把 Vulkan 着色器编译成跨平台的 SPIR-V 二进制发布，等部署到游戏玩家电脑上后，然后再输入显卡驱动中的 LLVM 得到 LLVM IR 后优化，编译生成最适合当前玩家显卡体质的 GPU 汇编。

#### IR 的二进制压缩版：字节码

字节码和 IR 的关系，正如汇编语言和机器二进制码的关系，之间是一一对应的翻译关系。只不过字节码是压缩的，对计算机友好；而 IR 是人类可读的 ASCII 字符，方便人类阅读和调试。

 > {{ icon.tip }} 注意字节码和机器码不同，他依然属于中间表示（只不过是压缩得人类看不懂的高效二进制版 IR），并不能直接在计算机中执行，LLVM 字节码只能在 lli 虚拟机中解释执行。

 > {{ icon.story }} 但和 Java 的字节码又不一样，LLVM 的字节码本来就是二进制的 IR。而 IR 并不跨平台，所以字节码也不跨平台。LLVM 团队提供 lli 工具主要是为了方便临时测试 IR，用于生产环境的肯定还是 llc 编译好产生真正的高效机器码。

#### 汇编语言（ASM）

#### 汇编语言的终局：机器码

### LLVM IR

LLVM IR 完全文档：https://llvm.org/docs/LangRef.html

> {{ icon.tip }} 不建议按顺序全部阅读完，这么多文档小彭老师都看不完。建议遇到了不熟悉的指令时，再去针对性地找到相应章节，学习
