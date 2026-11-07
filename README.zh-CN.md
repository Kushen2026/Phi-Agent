# Phi

一个面向 Windows 的本地优先桌面智能体，用 C++20 编写，自带一套用于图像、
视频、音乐与语音生成的 CUDA 推理引擎。

Phi 把两样东西装进同一个二进制：

* **智能体外壳** —— 一个 WebView2 聊天窗口，承载 LLM 会话：可以读写编辑
  文件、执行 shell 命令、搜索网页，并把任务派给后台子代理；
* **媒体引擎** —— 一个节点图运行时，在本机 NVIDIA GPU 上跑 Qwen-Image-2.1
  （图像）、MiniMax H3（视频 + 音频）、ACE-Step 1.5（音乐）和 Breeze-TTS-2
  （语音合成），并以四个工具的形式交给智能体。

一切都在本机运行。除了你自己配置的 LLM API，不涉及任何云服务。

---

## 目录

1. [它能做什么](#它能做什么)
2. [环境要求](#环境要求)
3. [仓库结构](#仓库结构)
4. [构建](#构建)
5. [运行](#运行)
6. [配置](#配置)
7. [智能体](#智能体)
8. [媒体引擎](#媒体引擎)
9. [命令行接口](#命令行接口)
10. [数据目录](#数据目录)
11. [诊断](#诊断)
12. [测试](#测试)
13. [安全须知](#安全须知)
14. [许可](#许可)

---

## 它能做什么

### 智能体

一个聊天窗口，包含会话记录、工作目录选择、模型 / 思考级别控制、技能面板、
斜杠命令面板和设置对话框。智能体循环是一个严格的两阶段步骤：

```
THINK    流式接收一次模型响应（思考 + 工具调用 + 可选的正文）
EXECUTE  把这次响应里的所有工具调用当作一批执行，把结果追加回历史
          … 重复，直到模型给出不带工具调用的最终回答
```

内置工具（每一个都能在「设置 → 工具设置」里开关、设定权限范围与输出上限）：

| 工具 | 作用 |
| --- | --- |
| `bash` | 执行 shell 命令，实时输出，并对结果做压缩整理 |
| `read` | 读取文本文件；图片 / 视频则作为多模态内容交给模型 |
| `write` | 创建新文件或覆盖现有文件 |
| `edit` | 按精确文本替换，一组替换批量执行 |
| `ls` | 列出目录，可选递归 |
| `grep` | 正则搜索文件内容（优先用自带的 `rg.exe`，不可用时走原生实现） |
| `web_search` | Bing 搜索，解析成带摘要和网址的结果列表 |
| `fetch` | 抓取某个网址并把页面转成文本 |
| `agent` | 启动后台子代理，完成后自动回传摘要 |
| `image_generate` | Qwen-Image-2.1 → PNG（文生图或参考图编辑） |
| `video_generate` | MiniMax H3 ref2va → MP4（自带生成音频） |
| `music_generate` | ACE-Step 1.5 → WAV（风格标签 + 歌词） |
| `tts_speak` | Breeze-TTS-2 → WAV（凭音色描述做音色设计） |

提示词与工具 schema 都有中文和英文两份，由
`settings.json → systemPromptLanguage` 选择。

### 媒体引擎

四条现成的链，全部由同一套节点注册表搭成：

```
图像:  ModelLoader ─┐
       ClipLoader ──┼─ QwenImageEditEncode ─┬─ CONDITIONING ─┐
       VaeLoader ───┘                      └─ LATENT ───────┼─ Sampler ─ VaeDecode ─ SaveImage
       EmptyLatent ────────────────────────────────────────┘      ↑
       Scheduler(model) ──────────────────────────────────────────┘

视频:  ModelLoader ─┬─ H3SigmaShift ──────────┐
       ClipLoader ──┴─ H3Ref2VaEncode ─┐     ├─ Sampler ─┬─ VaeDecode(video) ─┐
       VaeLoader(video) ───────────────┘               └─ VaeDecode(audio) ───┼─ SaveVideo
       VaeLoader(audio) ──────────────────────────────┘                      │
       EmptyLatentVideo ──────────────────────────────┘                      │
       Scheduler(model) ──────────────────────────────┘                      │

音乐:  ModelLoader(ace_step15) ─┬─ AceTextEncode ─┐
       ClipLoader(ace15, 双塔) ─┘                 │
       VaeLoader(ace_audio) ────────────┐         ├─ Sampler ─ VaeDecode ─ SaveAudio
       AceModelSampling ─ Scheduler ────┴─────────┘
       EmptyLatent(秒数) ───────────────┘

语音:  ModelLoader(breeze_tts) ─ BreezeTextEncode ─ BreezeSampler ─ BreezeDecode ─ SaveAudio
```

这张图是 ComfyUI 风格的强类型 socket 系统（`core/graph/`）：节点只注册一次
（`core/nodes/`），链是构造函数（`core/workflows/`），执行器按依赖顺序遍历。
加一个新能力意味着写一个节点并接上线，而不是重写整条流水线。

引擎刻意做到自包含：

* **构建期不需要 CUDA SDK。** 内核是 CUDA C++ 字符串常量，由 NVRTC（自带的
  `nvrtc64_*.dll`）在运行时针对检测到的 GPU 编译，产出的 cubin 缓存到
  `data/media_shader_cache/`。
* **运行期不需要 Python、PyTorch 或 ComfyUI。** 权重直接从 `.safetensors`
  读取（内存映射，按需流式送入显存）。
* **checkpoint 存什么精度就读什么精度**，不做转换：`int8_tensorwise`
  （含 256 宽的 convrot / Hadamard 旋转）、`asym_w4a8_int8`、`w6a8_int8`、
  `convrot_w4a4`、`nvfp4` / `mxfp4`、`fp8`（E4M3 / E5M2），以及稠密的
  `f16` / `bf16` / `f32`。GGUF 会被点名拒绝。
* **只有一个显存记账员。** 每一块常驻字节在提交前先记账；超预算的开销是被
  **拒绝**，绝不悄悄缩水。上限由显卡本身与该进程当前的驱动预算共同推导。
* **主机侧分层。** 放不进显存的权重留在锁页（或普通）内存里流式读取，配有
  前瞻预取槽；一个进程级的账本决定哪些张量值得锁页。

---

## 环境要求

**平台**

* Windows 10 或更新版本，x64。构建面向 Win10 API（`_WIN32_WINNT=0x0A00`），
  并链接 Media Foundation、DXGI、DWM 与 WebView2。
* 必须安装 WebView2 Runtime（Evergreen）。Windows 11 与较新的 Windows 10
  已内置；否则请从微软官网安装。

**GPU**

* NVIDIA 显卡 + 较新的驱动（`nvcuda.dll` 由驱动提供）。
* 计算能力 6.0 及以上（`int8` `dp4a` 路径）；内核按当前显卡的
  `sm_<major><minor>` 精确编译。
* 显存：引擎会流式读取权重，图像链在 6 GB 级别的卡上可以跑。视频链的 DiT
  在磁盘上就有约 21 GB，在工作站级显卡以下基本是每个采样步重读一遍。视频
  模型对张量核（`sm_70+`）依赖很重。

**构建机**

* 除了仓库本身什么都不需要：GCC 16（MinGW-w64、UCRT）、mingw32-make、
  windres、`rg.exe` 全都在 `vendor/` 下。
* makefile 的 recipe 是 POSIX 语法（`mkdir -p`、`cp`、`rm -rf`），请在
  Git Bash / MSYS2 / WSL 里执行。`./make.sh` 是一个薄包装：把 `TMPDIR`
  指到 `C:\Temp` 后 exec `vendor/gcc/bin/mingw32-make.exe`。

**磁盘**

* 四个模型家族合计约 80 GB。每条链的文件都是单独选择的（见
  [配置](#配置)），所以可以只装你要用的。

---

## 仓库结构

```
app/                桌面外壳：wWinMain、Win32 窗口、WebView2 引导、开屏
                    画面、目录选择对话框（`shell.cpp`）
assets/             phi.ico
bin/                构建产物：phi.exe + webview2loader.dll + nvrtc DLL
core/               引擎本体（18 个分组，一层嵌套）
  util/             基础字符串/路径/文件、JSON、rtk 输出压缩、utf8、i18n、
                    共用的媒体原语
  net/              回环 HTTP/1.1 + WebSocket 服务器，WinHTTP 客户端
  llm/              提供方客户端（anthropic-messages、openai-completions）、
                    模型目录
  store/            settings.json、webui-configs.json、会话 .jsonl 存档、
                    技能与提示词模板扫描
  agent/            智能体循环、会话、子代理、系统提示词、上下文 token 估算
  tools/            内置工具与四个媒体工具
  ui/               协议服务器（UI 桥）与主题解析
  graph/            节点图框架：socket、注册表、执行器，以及公开的
                    workflow 头文件
  nodes/            一个节点一个文件 —— 这就是节点目录
  workflows/        一条现成链一个文件（image / video / music / tts / audio）
  runtime/          CUDA 上下文与设备、NVRTC 编译与 cubin 缓存、派发、
                    调度、显存预算
  host/             safetensors 读取、量化与反量化、主机侧分层账本
                    （锁页 / 前瞻 / 页缓存）
  models/           各模型的前向（DiT、VAE、文本编码器）
  sampling/         采样器与 sigma 调度器（与 ComfyUI 兼容）
  io/               图像（WIC）、音频（Media Foundation）、视频（MF
                    SourceReader）以及 MP4 封装
  text/             BPE 分词器与视觉预处理
  bench/            `--media-bench` 验收工具
  kernels/          CUDA 内核源码（唯一跑 GPU 路径的代码）
ui/                 index.html、app.js、styles.css —— WebView2 前端
data/               运行时状态（设置、会话、分词器文件、缓存）
models/             checkpoint，每条链一个目录
vendor/             gcc/（MinGW-w64 工具链 + make + rg）、cuda/（nvrtc DLL
                    与头文件）、webview2/（loader DLL 与头文件）
makefile            整个构建过程
make.sh             调用自带 make 的 POSIX shell 包装
```

大致规模：`core/` 约 6.5 万行 C++，`ui/` 约 8 千行 JS/CSS/HTML，
`app/` 约 500 行。

---

## 构建

```bash
./make.sh            # 等价于 vendor/gcc/bin/mingw32-make.exe all
```

在仓库根目录的 POSIX shell 里执行。默认目标链接出 `bin/phi.exe`
（GUI 子系统，`-municode -mwindows`，静态链接），并把
`webview2loader.dll` 与 `nvrtc*` 系列 DLL 从 `vendor/` 复制进 `bin/`。

常用变体：

```bash
./make.sh OBJDIR=build/obj/w1     # 隔离目标文件（并行跑两份构建）
./make.sh clean                   # 删除 build/ 与 bin/
./make.sh -j8                     # 并行编译
```

说明：

* 目标文件路径镜像源码树（`build/obj/core/io/image_io.o`），并通过
  `-MMD -MP` 生成的依赖文件回读，所以改一个头文件会重建所有包含它的
  编译单元。
* 构建不需要 CUDA SDK：`core/runtime/cuda_api.cpp` 自己声明所需的
  运行时 / NVRTC 入口并动态加载 DLL。
* `windres` 把 `app/phi.rc`（图标）编进可执行文件。

---

## 运行

```bash
bin/phi.exe
```

窗口立刻出现并绘制带主题的 Φ 开屏画面，同时 WebView2 运行时与回环协议
服务器在后台启动；WebView 首帧就绪后接管画面。服务器绑定一个随机的回环
端口，WebView 导航到它 —— 前后端之间从不经过网络。

`phi.exe` 是 GUI 子系统二进制。带控制台子命令（`--media-bench`、
`--media-tool`）启动时，它会附着到父控制台（没有就分配一个），把纯文本
输出到 stdout / stderr。

首次启动按需创建 `data/`（设置、会话、WebView 配置、着色器缓存）。

---

## 配置

所有状态都是 `data/` 下的 JSON。

### `data/settings.json`

```jsonc
{
  "hotkeys":   { "input.send": "Enter", "input.newline": "Shift+Enter",
                 "session.new": "Ctrl+Alt+N", "input.focus": "Ctrl+I" },
  "appearance":{ "theme": "mint", "radius": "none", "density": "compact", … },
  "tools": {
    "enabled":     { "bash": true, "read": true, …, "tts_speak": true },
    "permissions": { "bash": "everywhere", "read": "everywhere", … },  // 或 "in-cwd"
    "outputLimits":{ "bash": 120000, "read": 120000 },
    "subagent":    { "tools": ["bash","read","write","edit"], "thinking": "…" },
    "media": {
      "image_dit": "qwen_image_2.1/qwen_image_2.1_int8_convrot.safetensors",
      "image_te":  "qwen_image_2.1/qwen3vl_8b_int8_convrot.safetensors",
      "image_vae": "qwen_image_2.1/qwen_image_2.1_vae_bf16.safetensors",
      "video_dit": "minimax_h3/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors",
      "video_te":  "minimax_h3/minimax_h3_qwen3vl_32b_convrot_int4.safetensors",
      "video_vae": "minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors",
      "video_avae":"minimax_h3/minimax_h3_audio_vae_fp32.safetensors",
      "music_dit": "ace_step_1.5/acestep_v1.5_xl_sft_bf16.safetensors",
      "music_te":  "ace_step_1.5/qwen_0.6b_ace15.safetensors",
      "music_lm":  "ace_step_1.5/qwen_4b_ace15.safetensors",
      "music_vae": "ace_step_1.5/ace_1.5_vae.safetensors",
      "tts_model": "breeze_tts_2/breeze_tts_2_bf16.safetensors",
      "tts_codec": "breeze_tts_2/breeze_tts_2_audio_tokenizer.safetensors",
      "image_steps": 25, "video_steps": 20, "music_steps": 50,
      "image_sampler": "euler", "image_scheduler": "simple",
      "video_sampler": "euler", "video_scheduler": "beta",
      "music_sampler": "euler", "music_scheduler": "simple",
      "image_loras": [], "image_lora_strengths": [1.0],
      "video_loras": [], "video_lora_strengths": [1.0]
    }
  },
  "defaultProvider": "…", "defaultModel": "…", "defaultThinkingLevel": "max",
  "systemPromptLanguage": "zh"
}
```

以上内容都能在「设置」（API 配置 · 工具设置 · 快捷键 · 界面风格 · 通用 ·
关于）里改，文件本身是同一份数据。

### `data/models.json` 与 `data/webui-configs.json`

提供方有两个来源：`assets/models-catalog.json`（随程序分发的已知提供方目录，
只读）和 `models.json`（你自己添加的）。界面的 API 配置页把「配置档」
写进 `webui-configs.json`：

```json
{
  "active": "My Provider",
  "profiles": [{
    "name": "My Provider",
    "model": "…", "baseUrl": "https://…/v1", "apiKey": "…",
    "apiFormat": "openai-completions",      // 或 "anthropic-messages"
    "subagent": { "model": "…", "baseUrl": "…", "apiKey": "…", "apiFormat": "…" },
    "vision": "image",                       // "none" | "image" | "image_video"
    "systemPromptLanguage": "zh"
  }]
}
```

`vision` 决定 `read` 工具能不能把一张图片（或整段视频）交给模型。
纯文本模型会得到一句明确的拒绝，而不是被端点打回的内容。思考级别为
`off` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`
（按提供方映射）。

API 密钥的解析顺序：配置档显式密钥 → `models.json` 提供方密钥 → 目录内置
密钥 → 环境变量。

### 模型

把 checkpoint 放到 `models/` 下，每条链一个目录（扁平目录也能识别）。
引擎会从 `data/` 和会话工作目录向上搜索，寻找形似 checkpoint 目录的
`models/`。

分词器文件放在 `data/tokenizers/`，随程序走而不是随模型目录走：默认是
`qwen2.5/vocab.json`、`merges.txt`、`tokenizer_config.json` 与
`breeze_tts_2/tokenizer.json`，每条链都可以单独改。

---

## 智能体

* **会话**：一个会话对应 `data/sessions/<bucket>/` 下的一个 `.jsonl`
  文件，bucket 由工作目录推导。每次落盘都是「写临时文件 + 原子 rename」的
  整体重写，所以崩溃不可能留下半行 JSON；回放时还会对旧追加式格式的遗留问题
  做一次性自愈。
* **上下文压缩**：上下文窗口将满时把较早的对话总结掉。token 估算器（用真实
  BPE 词表标定）驱动上下文进度条与压缩报告。`/compact` 可手动触发。
* **子代理**（`agent` 工具）在后台跑，有自己的一套工具与思考级别，完成后把
  摘要回灌到父会话。
* **技能**是带 YAML frontmatter 的 `SKILL.md`，在 `data/skills/`（全局）
  和 `<cwd>/data/skills/`（项目）里发现。
* **提示词模板**是 `data/prompts/` 与 `<cwd>/.phi/prompts/` 下的 markdown
  文件，会出现在 `/` 命令面板里，和内建的 `/copy`、`/session`、
  `/compact` 并列。
* **附件**：把文件拖进输入框会以 base64 上传到 `data/uploads/` 并按路径
  引用 —— 会话文件里存的是路径，不是几 MB 的 base64。

默认快捷键：`Enter` 发送 · `Shift+Enter` 换行 · `Ctrl+Alt+N` 新会话 ·
`Ctrl+I` 聚焦输入框 —— 这四组可在「设置 → 快捷键」里改。
`Ctrl+Shift+C`（设置）与 `Ctrl+B`（技能面板）是固定快捷键。

---

## 媒体引擎

### 工具

每个媒体工具都只是某一张图的一层薄壳。它们共用一个进程级引擎（一个 CUDA
上下文，同一时刻只跑一次生成），首次使用时惰性打开，之后复用。

**`image_generate`** —— Qwen-Image-2.1，输出 PNG。

| 参数 | 说明 |
| --- | --- |
| `prompt` | 英文，100–300 词（主体 / 环境 / 风格 / 镜头 / 氛围 / 细节） |
| `width`、`height` | 400–2000 像素，画布上取 32 的倍数，生成后再裁回精确尺寸 |
| `ref_images` | 0–9 个绝对路径；带参考图时这条链变成参考隐空间的编辑 |
| `output` | `.png` 的绝对路径 |

**`video_generate`** —— MiniMax H3 ref2va，输出 MP4（H.264 + AAC）。

| 参数 | 说明 |
| --- | --- |
| `prompt` | 英文，300–600 词：先用 `<Subject N>` / `<Picture N>` / `<Audio N>` 声明主体与引用，再按镜头顺序（`[Shot N] At SS.s`）写，最后是场景与音乐 |
| `duration_frames` | 5–362 帧、24 fps（会被对齐到合法的 `17k+5` 时间网格） |
| `width`、`height` | 128–2048 像素，取偶数值（yuv420），画布上对齐到 32 |
| `ref_images` | 0–9 张，严格对应 `<Picture N>` 顺序 |
| `ref_audios` | 0–3 段，严格对应 `<Audio N>` 顺序 |
| `output` | `.mp4` 的绝对路径 |

**`music_generate`** —— ACE-Step 1.5，输出 48 kHz WAV。

| 参数 | 说明 |
| --- | --- |
| `tags` | 英文风格标签，逗号分隔 |
| `lyrics` | 歌词，用 `[verse]` / `[chorus]` 分行；纯器乐写 `[instrumental]` |
| `duration` | 时长（秒） |
| `bpm`、`language`、`keyscale`、`timesignature` | 默认 72 / `en` / `D minor` / `4` |
| `output` | `.wav` 的绝对路径 |

**`tts_speak`** —— Breeze-TTS-2 音色设计，输出 24 kHz WAV。不接受参考音频：
音色完全来自 `instruction` 的描述。

| 参数 | 说明 |
| --- | --- |
| `text` | 要朗读的文本（中英双语，可内联 `[笑]` / `(sigh)` 之类的语气事件） |
| `instruction` | 音色描述（必填） |
| `output` | `.wav` 的绝对路径 |

种子由工具每次随机抽取并在报告里给出，不能外部指定。

### 采样

步数来自工具设置（默认图像 25、视频 20、音乐 50）。

采样器：`euler`、`euler_ancestral`、`heun`、`dpmpp_2m`、
`dpmpp_2s_ancestral`、`dpmpp_2m_sde`、`dpmpp_sde`、`res_multistep`、
`lcm`。
调度器：`simple`、`normal`、`sgm_uniform`、`karras`、`exponential`、
`ddim_uniform`、`beta`、`linear_quadratic`、`kl_optimal`。

默认值复刻了原始发布工作流（图像 `euler`+`simple`，视频
`res_multistep`+`beta`，音乐 `euler`+`simple`）。采样器与调度器的数学
是 ComfyUI 实现的移植，由 makefile 里的主机测试钉住。

### 节点目录

`MediaModelLoader`、`MediaClipLoader`、`MediaVaeLoader`、
`MediaLoraLoader`、`MediaQwenImageEditEncode`、`MediaH3Ref2VaEncode`、
`MediaH3SigmaShift`、`MediaQwenImage21Cache`、`MediaEmptyLatent`、
`MediaScheduler`、`MediaSampler`、`MediaVaeDecode`、
`MediaVaeEncodeImage`、`MediaLoadImage`、`MediaSaveImage`、
`MediaSaveAudio`、`MediaSaveVideo`、`MediaSeed`、`MediaConstInt`、
`MediaAceTextEncode`、`MediaAceModelSampling`、`MediaBreezeTextEncode`、
`MediaBreezeSampler`、`MediaBreezeDecode`。

一个节点对应一个**阶段**，而不是一个模型：`MediaSampler` 按模型的架构
分派，`MediaVaeDecode` 按 VAE 的角色分派，`MediaEmptyLatent` 按它自己的
`arch` 参数分派。

---

## 命令行接口

```bash
phi.exe --media-bench [选项]      # 环境、显存预算、流式吞吐、int8 自检、
                                  # 内核基准，可选跑真实链
phi.exe --media-tool <名称> …     # 按智能体的方式运行某一个媒体工具
```

`--media-bench` 的选项（节选）：

```
[--quick] [--verbose] [--json out.json] [--model path]...
[--image] [--image-size WxH] [--image-prompt text] [--image-out out.png]
[--image-seed N] [--image-ref path]... [--image-ref-area px2]
[--image-negative text] [--image-cfg F]
[--video] [--video-size WxH] [--video-frames N] [--video-prompt text]
[--video-out out.mp4] [--video-seed N] [--video-steps N]
[--video-ref path]... [--video-ref-clip path]... [--video-ref-clip-audio path]...
[--video-ref-audio path]... [--video-ref-image-size match|max]
[--video-lora path]... [--video-lora-strength F]...
```

任一硬性检查失败都会返回非零退出码，所以它可以用来卡 CI。

`--media-tool` 在工具名之后接受：

* `schema` —— 打印工具的 JSON schema（按「全能模型」视角）；
* `graph` —— 构建该链的图并对着注册表校验每个节点、端口和连线，不需要
  GPU、不需要 checkpoint；
* `resolve` —— 打印当前 `settings.json` 下每个模型角色解析到哪个文件、
  该文件是否存在；
* 一个 JSON 参数对象 —— 真正运行该工具（同一个 `ToolDef::run`、同一个
  `ToolContext`，只是没有模型在回路里）。

```bash
phi.exe --media-tool image_generate "{\"prompt\":\"a red fox in tall grass\",\"width\":1024,\"height\":1024,\"output\":\"C:/out/fox.png\"}"
phi.exe --media-tool video_generate schema
phi.exe --media-tool resolve
```

---

## 数据目录

| 路径 | 内容 |
| --- | --- |
| `data/settings.json` | 快捷键、外观、工具设置、媒体角色、默认值 |
| `data/models.json` | 用户添加的提供方与模型 |
| `data/webui-configs.json` | 设置对话框写出的 API 配置档 |
| `data/sessions/<bucket>/*.jsonl` | 一个文件就是一份自洽的会话快照 |
| `data/skills/`、`data/prompts/` | 全局技能与提示词模板 |
| `data/tokenizers/` | BPE 词表/合并表/配置，以及 Breeze-TTS-2 的 tokenizer.json |
| `data/uploads/` | 输入框里附带的文件 |
| `data/media_shader_cache/` | 编译好的 cubin，按源码/入口/架构/NVRTC/头文件路径/快速数学 分键 |
| `data/webview/` | WebView2 的用户数据目录 |

`data/` 按需创建；除你想保留的 JSON 之外，删除是安全的。

---

## 诊断

引擎的诊断埋在环境变量后面，默认全关（未设置时零开销）。值得知道的几个：

| 变量 | 作用 |
| --- | --- |
| `PHI_TRACE` | 启动 / 传输 / 会话的时间戳追踪，输出到 stderr |
| `PHI_MEDIA_TRACE` | 刚跑完那张图的逐节点耗时 |
| `PHI_MEDIA_STATE` | 一次生成边界上进程仍持有什么 |
| `PHI_VRAM_BUDGET_MB` | 覆盖显存上限（仍受物理显存约束） |
| `PHI_VRAM_MARGIN_PERCENT` | 覆盖预算余量（限制在 50–100） |
| `PHI_RING_MB` | 钉住上传环大小（用来刻意走小环路径） |
| `PHI_CUDA_BIN`、`PHI_CUDA_INCLUDE` | 额外搜索 NVRTC DLL / 头文件的根目录 |
| `PHI_TOKENIZER_DIR` | 一次性覆盖所有分词器角色 |
| `PHI_W4A8_CPU` | 强制走主机解码路径（分组尺度权重的 A/B） |
| `PHI_PERF` | 以 `?perf=1` 打开界面 |
| `PHI_IMG_STEPS`、`PHI_VIDEO_STEPS` | 单次基准的步数覆盖 |

还有更多，`grep getenv core/` 全部可见；绝大多数是内核调优用的单点实验开关。

---

## 测试

makefile 定义了若干主机侧测试目标，用同一套自带工具链把少量 `core/` 源码
链接成控制台程序：

```bash
./make.sh precision-test     # 每个加载器的每一种权重量族，对着参考值
./make.sh sampling-test      # 采样器/调度器数学，不需要设备
./make.sh workflow-test      # 工具设置 → workflow → 节点的接缝
./make.sh token-test         # 上下文 token 估算器 vs 真实 BPE 词表
```

`grouped-decode-test` 还需要 `bin/phi.exe` 和 GPU，因为它要把分组尺度权重的
**设备**解码与主机解码逐字节对比。

> 注意：这些目标引用了 `tests/` 下的源码（以及
> `tests/data/*.safetensors`），而它们不在本快照里 —— 目标是定义好的，
> 夹具没有随包提供。要用之前先把它们补进你的工作树。

---

## 安全须知

* UI 桥只绑定 `127.0.0.1`，且每个请求都校验 `Host` 头，所以网页无法驱动
  这个本地服务。
* `bash` 会执行模型让它执行的任何东西。每个工具的权限范围
  （`everywhere` / `in-cwd`）和「设置 → 工具设置」里的开关就是护栏，
  请有意识地设置。
* API 密钥以**明文**存放在 `data/models.json` 与 `data/webui-configs.json`，
  请把这两个文件当机密对待。
* 会话会原样记录工具参数与输出，包括智能体读过的文件内容。`data/sessions/`
  的敏感程度等同于你让它工作的那些目录。

---

## 许可

本源码树没有附带 `LICENSE` 文件。在补充之前，请视其代码为作者的作品：
未经许可不得再分发或商用。`vendor/` 下的第三方组件（MinGW-w64 GCC、
CUDA NVRTC 运行时与头文件、WebView2 loader、ripgrep）各自遵循自己的许可。
