# Phi

A local-first desktop agent for Windows, written in C++20, that ships its own
CUDA inference engine for image, video, music and speech generation.

Phi is two things in one binary:

* **an agent shell** — a chat window (WebView2) hosting an LLM session that can
  read, write and edit files, run shell commands, search the web, and delegate
  to background subagents;
* **a media engine** — a node-graph runtime that runs Qwen-Image-2.1 (image),
  MiniMax H3 (video + audio), ACE-Step 1.5 (music) and Breeze-TTS-2
  (text-to-speech) locally on an NVIDIA GPU, exposed to the agent as four tools.

Everything runs on the machine. No cloud services are involved beyond the LLM
API you configure yourself.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Repository layout](#repository-layout)
4. [Build](#build)
5. [Run](#run)
6. [Configuration](#configuration)
7. [The agent](#the-agent)
8. [The media engine](#the-media-engine)
9. [Command-line interface](#command-line-interface)
10. [Data directories](#data-directories)
11. [Diagnostics](#diagnostics)
12. [Tests](#tests)
13. [Security notes](#security-notes)
14. [License](#license)

---

## What it does

### Agent

A single chat window with a session transcript, a working-directory selector,
model/thinking-level controls, a skills panel, a slash-command palette and a
settings dialog. The agent loop is a strict two-phase step:

```
THINK    stream one model response (reasoning + tool calls + optional text)
EXECUTE  run every tool call from that response as a batch, append the results
         … repeat until the model produces a final answer with no tool calls
```

Built-in tools (each can be enabled, given a permission scope and an output
limit in Settings → Tool settings):

| Tool | Purpose |
| --- | --- |
| `bash` | Run a shell command, stream its output, compact the result |
| `read` | Read a text file, or hand an image/video to the model as multimodal content |
| `write` | Create or overwrite a file |
| `edit` | Exact-string replacements, applied as a batch |
| `ls` | List a directory, optionally recursive |
| `grep` | Regex content search (uses the vendored `rg.exe`, with a native fallback) |
| `web_search` | Bing search, parsed into results with snippets and URLs |
| `fetch` | Fetch a URL and convert the page to text |
| `agent` | Start a background subagent that reports back a summary |
| `image_generate` | Qwen-Image-2.1 → PNG (text-to-image or reference-image editing) |
| `video_generate` | MiniMax H3 ref2va → MP4 (with generated audio) |
| `music_generate` | ACE-Step 1.5 → WAV (tags + lyrics) |
| `tts_speak` | Breeze-TTS-2 → WAV (voice design from a timbre description) |

Prompt strings and tool schemas exist in Chinese and English, driven by
`settings.json → systemPromptLanguage`.

### Media engine

Four ready-made chains, all built from the same registry of nodes:

```
image:  ModelLoader ─┐
        ClipLoader ──┼─ QwenImageEditEncode ─┬─ CONDITIONING ─┐
        VaeLoader ───┘                      └─ LATENT ───────┼─ Sampler ─ VaeDecode ─ SaveImage
        EmptyLatent ────────────────────────────────────────┘      ↑
        Scheduler(model) ──────────────────────────────────────────┘

video:  ModelLoader ─┬─ H3SigmaShift ──────────┐
        ClipLoader ──┴─ H3Ref2VaEncode ─┐     ├─ Sampler ─┬─ VaeDecode(video) ─┐
        VaeLoader(video) ───────────────┘               └─ VaeDecode(audio) ───┼─ SaveVideo
        VaeLoader(audio) ──────────────────────────────┘                      │
        EmptyLatentVideo ──────────────────────────────┘                      │
        Scheduler(model) ──────────────────────────────┘                      │

music:  ModelLoader(ace_step15) ─┬─ AceTextEncode ─┐
        ClipLoader(ace15, dual) ─┘                 │
        VaeLoader(ace_audio) ────────────┐         ├─ Sampler ─ VaeDecode ─ SaveAudio
        AceModelSampling ─ Scheduler ────┴─────────┘
        EmptyLatent(seconds) ────────────┘

tts:    ModelLoader(breeze_tts) ─ BreezeTextEncode ─ BreezeSampler ─ BreezeDecode ─ SaveAudio
```

The graph is a ComfyUI-style typed-socket system (`core/graph/`): nodes are
registered once (`core/nodes/`), the chains are constructor functions
(`core/workflows/`), and the executor walks the graph in dependency order.
Adding a capability means writing one node and wiring it — not rewriting a
pipeline.

The engine is deliberately self-contained:

* **No CUDA SDK at build time.** Kernels are CUDA C++ string literals; NVRTC
  (vendored `nvrtc64_*.dll`) compiles them for the detected GPU at runtime,
  and the resulting cubins are cached under `data/media_shader_cache/`.
* **No Python, no PyTorch, no ComfyUI at runtime.** Weights are read straight
  from `.safetensors` files (memory-mapped, streamed to VRAM on demand).
* **Any weight precision the checkpoint stores** is read verbatim:
  `int8_tensorwise` (with the 256-wide convrot/Hadamard rotation),
  `asym_w4a8_int8`, `w6a8_int8`, `convrot_w4a4`, `nvfp4`/`mxfp4`,
  `fp8` (E4M3/E5M2), and dense `f16`/`bf16`/`f32`. GGUF is refused by name.
* **One VRAM accountant.** Every resident byte is charged before it is
  committed; a charge that would exceed the budget is refused, never silently
  truncated. The ceiling is derived from the card and the driver's live budget.
* **Host-side tiers.** Weights that do not fit in VRAM are held in pinned (or
  plain) system memory and streamed, with a look-ahead prefetch slot; a
  process-wide ledger decides which tensors are worth pinning.

---

## Requirements

**Platform**

* Windows 10 or newer, x64. The build targets the Win10 APIs
  (`_WIN32_WINNT=0x0A00`) and links the Media Foundation, DXGI, DWM and
  WebView2 surfaces.
* The WebView2 Runtime (Evergreen) must be installed. It ships with Windows 11
  and recent Windows 10 builds; otherwise install it from Microsoft.

**GPU**

* An NVIDIA GPU with a current driver (`nvcuda.dll` comes from the driver).
* Compute capability 6.0 or better (`int8` `dp4a` path); the kernels are
  compiled for exactly the `sm_<major><minor>` of the card present.
* VRAM: the engine streams weights, so it runs on a 6 GB class card for the
  image chain. The video chain's DiT alone is ~21 GB on disk — expect it to be
  streamed per sampling step on anything below a workstation card. The video
  model also benefits strongly from tensor cores (`sm_70+`).

**Build host**

* Nothing beyond the repository itself: GCC 16 (MinGW-w64, UCRT) mingw32-make,
  windres and `rg.exe` all live under `vendor/`.
* The makefile's recipes are POSIX (`mkdir -p`, `cp`, `rm -rf`), so run them
  from Git Bash, MSYS2 or WSL. `./make.sh` is a thin wrapper that execs
  `vendor/gcc/bin/mingw32-make.exe` with `TMPDIR` pointed at `C:\Temp`.

**Disk**

* The four model families together are ~80 GB as shipped. Each chain's files
  are selected individually (see [Configuration](#configuration)), so you can
  install only what you use.

---

## Repository layout

```
app/                the desktop shell: wWinMain, the Win32 window, WebView2
                    bootstrap, the splash, the folder picker (`shell.cpp`)
assets/             phi.ico
bin/                build output: phi.exe + webview2loader.dll + nvrtc DLLs
core/               the engine (18 groups, one level of nesting)
  util/             base strings/paths/files, JSON, the rtk output compactor,
                    utf8, i18n, shared media primitives
  net/              loopback HTTP/1.1 + WebSocket server, WinHTTP client
  llm/              provider clients (anthropic-messages, openai-completions),
                    the model catalog
  store/            settings.json, webui-configs.json, session .jsonl archive,
                    skills/prompt-template scanning
  agent/            the agent loop, sessions, subagents, system prompt,
                    the context-token estimator
  tools/            the built-in tools and the four media tools
  ui/               the protocol server (the UI bridge) and theme resolution
  graph/            the node-graph framework: sockets, registry, executor,
                    plus the public workflow headers
  nodes/            one file per node — the catalogue
  workflows/        one file per ready-made chain (image/video/music/tts/audio)
  runtime/          CUDA context & device, NVRTC compilation and the cubin
                    cache, dispatch, scheduling, the VRAM budget
  host/             safetensors reader, quantisation/dequantisation, the
                    host-side tier ledger (pinned / look-ahead / page cache)
  models/           the model forward passes (DiTs, VAEs, text encoders)
  sampling/         samplers and sigma schedulers (ComfyUI-compatible)
  io/               image (WIC), audio (Media Foundation), video (MF
                    SourceReader) and the MP4 muxer
  text/             BPE tokenizers and vision prep
  bench/            the `--media-bench` acceptance instrument
  kernels/          CUDA kernel sources (the only GPU-path code)
ui/                 index.html, app.js, styles.css — the WebView2 front end
data/               runtime state (settings, sessions, tokenizer files, caches)
models/             the checkpoints, one directory per chain
vendor/             gcc/ (MinGW-w64 toolchain + make + rg), cuda/ (nvrtc DLLs
                    and headers), webview2/ (loader DLL and header)
makefile            the whole build
make.sh             POSIX-shell wrapper around the vendored make
```

Rough size: ~65 k lines of C++ in `core/`, ~8 k lines of JS/CSS/HTML in `ui/`,
~0.5 k lines in `app/`.

---

## Build

```bash
./make.sh            # == vendor/gcc/bin/mingw32-make.exe all
```

From a POSIX shell at the repository root. The default goal links
`bin/phi.exe` (GUI subsystem, `-municode -mwindows`, statically linked) and
copies `webview2loader.dll` and the `nvrtc*` DLLs out of `vendor/` into
`bin/`.

Useful variants:

```bash
./make.sh OBJDIR=build/obj/w1     # isolate objects (two builds at once)
./make.sh clean                   # remove build/ and bin/
./make.sh -j8                     # parallel compile
```

Notes:

* Objects mirror the source tree (`build/obj/core/io/image_io.o`), and
  `-MMD -MP` dependency files are pulled back in, so editing a header rebuilds
  every translation unit that includes it.
* The build needs no CUDA SDK: `core/runtime/cuda_api.cpp` declares the
  handful of runtime/NVRTC entry points itself and loads the DLLs dynamically.
* `windres` compiles `app/phi.rc` (the icon) into the executable.

---

## Run

```bash
bin/phi.exe
```

The window appears immediately and paints a themed Φ splash while the WebView2
runtime and the loopback protocol server start; the WebView takes over on its
first frame. The server binds a random loopback port and the WebView navigates
to it — the two halves never talk over the network.

`phi.exe` is a GUI-subsystem binary. When invoked with a console subcommand
(`--media-bench`, `--media-tool`) it attaches to the parent console (or
allocates one) and streams plain text to stdout/stderr.

On first start `data/` is created as needed (settings, sessions, WebView
profile, shader cache).

---

## Configuration

All state is JSON under `data/`.

### `data/settings.json`

```jsonc
{
  "hotkeys":   { "input.send": "Enter", "input.newline": "Shift+Enter",
                 "session.new": "Ctrl+Alt+N", "input.focus": "Ctrl+I" },
  "appearance":{ "theme": "mint", "radius": "none", "density": "compact", … },
  "tools": {
    "enabled":     { "bash": true, "read": true, …, "tts_speak": true },
    "permissions": { "bash": "everywhere", "read": "everywhere", … },  // or "in-cwd"
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

All of this is editable in **Settings** (API config · tool settings · hotkeys ·
appearance · general · about); the file is the same data.

### `data/models.json` and `data/webui-configs.json`

Two places a provider can come from: `assets/models-catalog.json` (a shipped
catalog of known providers, read-only) and `models.json` (providers you add).
The UI's API-config page writes *profiles* to `webui-configs.json`:

```json
{
  "active": "My Provider",
  "profiles": [{
    "name": "My Provider",
    "model": "…", "baseUrl": "https://…/v1", "apiKey": "…",
    "apiFormat": "openai-completions",      // or "anthropic-messages"
    "subagent": { "model": "…", "baseUrl": "…", "apiKey": "…", "apiFormat": "…" },
    "vision": "image",                       // "none" | "image" | "image_video"
    "systemPromptLanguage": "zh"
  }]
}
```

`vision` decides whether the `read` tool may hand an image — or a whole
video — to the model. A text-only model gets a plain refusal instead of
content its endpoint would reject. Thinking levels are `off`, `minimal`,
`low`, `medium`, `high`, `xhigh`, `max` (mapped per provider).

API keys are resolved in this order: an explicit profile key → `models.json`
provider key → the catalog's embedded key → an environment variable.

### Models

Put checkpoints under `models/`, one directory per chain (a flat directory
also works). The engine searches upward from `data/` and from the session's
working directory for a `models/` folder that plausibly holds checkpoints.

Tokenizer files live under `data/tokenizers/` and travel with the program
rather than with the checkpoints: `qwen2.5/vocab.json`, `merges.txt`,
`tokenizer_config.json` and `breeze_tts_2/tokenizer.json` by default, each
individually overridable per chain.

---

## The agent

* **Sessions** are one `.jsonl` file per session under
  `data/sessions/<bucket>/`, where the bucket is derived from the working
  directory. Each write is a whole-file rewrite to a temporary file followed by
  an atomic rename, so a crash can never leave a half-written JSON line; the
  replay self-heals leftovers from the old append-only format.
* **Context compaction** summarizes older turns when the context window
  fills. A token estimator (calibrated against the real BPE vocabulary) drives
  the meter and the compaction report. `/compact` forces it.
* **Subagents** (`agent` tool) run in the background on their own tool set and
  thinking level, and report a summary back into the parent transcript.
* **Skills** are `SKILL.md` files with YAML frontmatter, discovered in
  `data/skills/` (global) and `<cwd>/data/skills/` (project).
* **Prompt templates** are markdown files in `data/prompts/` and
  `<cwd>/.phi/prompts/`; they appear in the `/` palette next to the built-ins
  (`/copy`, `/session`, `/compact`).
* **Attachments**: dropping a file into the composer base64-uploads it to
  `data/uploads/` and references it by path — the session file stores the path,
  not megabytes of base64.

Default hotkeys: `Enter` send · `Shift+Enter` newline · `Ctrl+Alt+N` new
session · `Ctrl+I` focus input — all four rebindable in Settings → Hotkeys.
`Ctrl+Shift+C` (settings) and `Ctrl+B` (skills panel) are fixed.

---

## The media engine

### Tools

Each media tool is a thin surface over one graph. They share the process-wide
engine (one CUDA context, one generation at a time) which is opened lazily on
first use and reused afterwards.

**`image_generate`** — Qwen-Image-2.1, PNG out.

| Argument | Notes |
| --- | --- |
| `prompt` | English, 100–300 words (subject / environment / style / camera / mood / detail) |
| `width`, `height` | 400–2000 px, each a multiple of 32 on the canvas, cropped back to the exact request |
| `ref_images` | 0–9 absolute paths; with references the chain becomes a reference-latent edit |
| `output` | absolute `.png` path |

**`video_generate`** — MiniMax H3 ref2va, MP4 (H.264 + AAC) out.

| Argument | Notes |
| --- | --- |
| `prompt` | English, 300–600 words: `<Subject N>` / `<Picture N>` / `<Audio N>` declarations, then shots (`[Shot N] At SS.s`), scene, music |
| `duration_frames` | 5–362 frames at 24 fps (snapped to the legal `17k+5` temporal grid) |
| `width`, `height` | 128–2048 px, even values (yuv420), aligned to 32 on the canvas |
| `ref_images` | 0–9, in strict `<Picture N>` order |
| `ref_audios` | 0–3, in strict `<Audio N>` order |
| `output` | absolute `.mp4` path |

**`music_generate`** — ACE-Step 1.5, 48 kHz WAV out.

| Argument | Notes |
| --- | --- |
| `tags` | English style tags, comma separated |
| `lyrics` | lines with `[verse]` / `[chorus]` markers; `[instrumental]` for none |
| `duration` | seconds |
| `bpm`, `language`, `keyscale`, `timesignature` | defaults 72 / `en` / `D minor` / `4` |
| `output` | absolute `.wav` path |

**`tts_speak`** — Breeze-TTS-2 voice design, 24 kHz WAV out. No reference clip:
the voice comes from the `instruction` description.

| Argument | Notes |
| --- | --- |
| `text` | the words to read aloud (Chinese or English; inline `[笑]` / `(sigh)` events) |
| `instruction` | the timbre description (required) |
| `output` | absolute `.wav` path |

Seeds are drawn by the tool on every call and reported; you cannot pass one.

### Sampling

Steps come from tool settings (image 25, video 20, music 50 by default).

Samplers: `euler`, `euler_ancestral`, `heun`, `dpmpp_2m`,
`dpmpp_2s_ancestral`, `dpmpp_2m_sde`, `dpmpp_sde`, `res_multistep`,
`lcm`.
Schedulers: `simple`, `normal`, `sgm_uniform`, `karras`, `exponential`,
`ddim_uniform`, `beta`, `linear_quadratic`, `kl_optimal`.

Defaults reproduce the released workflows (image `euler`+`simple`, video
`res_multistep`+`beta`, music `euler`+`simple`). The scheduler and
sampler maths are ports of the ComfyUI implementations and are pinned by the
host tests in the makefile.

### Node catalogue

`MediaModelLoader`, `MediaClipLoader`, `MediaVaeLoader`, `MediaLoraLoader`,
`MediaQwenImageEditEncode`, `MediaH3Ref2VaEncode`, `MediaH3SigmaShift`,
`MediaQwenImage21Cache`, `MediaEmptyLatent`, `MediaScheduler`,
`MediaSampler`, `MediaVaeDecode`, `MediaVaeEncodeImage`, `MediaLoadImage`,
`MediaSaveImage`, `MediaSaveAudio`, `MediaSaveVideo`, `MediaSeed`,
`MediaConstInt`, `MediaAceTextEncode`, `MediaAceModelSampling`,
`MediaBreezeTextEncode`, `MediaBreezeSampler`, `MediaBreezeDecode`.

One node per *stage*, not per model: `MediaSampler` dispatches on the model's
architecture, `MediaVaeDecode` on the VAE's role, `MediaEmptyLatent` on its
own `arch` parameter.

---

## Command-line interface

```bash
phi.exe --media-bench [options]      # environment, VRAM budget, streaming
                                     # throughput, int8 self-test, kernel
                                     # benchmarks, optional real chain runs
phi.exe --media-tool <name> …        # run one media tool exactly as the agent does
```

`--media-bench` options (a subset):

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

It returns a non-zero exit code when a hard check fails, so it can gate CI.

`--media-tool` accepts, after the tool name:

* `schema` — print the tool's JSON schema (as a fully-capable model sees it);
* `graph` — build the chain's graph and validate every node, port and link
  against the registry, with no GPU and no checkpoint;
* `resolve` — print which file every model role resolves to for the current
  `settings.json`, and whether it exists;
* a JSON argument object — run the tool for real (same `ToolDef::run`, same
  `ToolContext`, no model in the loop).

```bash
phi.exe --media-tool image_generate "{\"prompt\":\"a red fox in tall grass\",\"width\":1024,\"height\":1024,\"output\":\"C:/out/fox.png\"}"
phi.exe --media-tool video_generate schema
phi.exe --media-tool resolve
```

---

## Data directories

| Path | Contents |
| --- | --- |
| `data/settings.json` | hotkeys, appearance, tool settings, media roles, defaults |
| `data/models.json` | user-added providers and models |
| `data/webui-configs.json` | API profiles written by the settings dialog |
| `data/sessions/<bucket>/*.jsonl` | one self-contained session snapshot per file |
| `data/skills/`, `data/prompts/` | global skills and prompt templates |
| `data/tokenizers/` | BPE vocab/merges/config, plus Breeze-TTS-2's tokenizer.json |
| `data/uploads/` | files attached in the composer |
| `data/media_shader_cache/` | compiled cubins, keyed by source/entry/arch/NVRTC/include/fast-math |
| `data/webview/` | the WebView2 user-data profile |

`data/` is created on demand and is safe to delete except for the JSON files
you want to keep.

---

## Diagnostics

The engine is instrumented behind environment variables, all off by default
(zero cost when unset). The ones worth knowing:

| Variable | Effect |
| --- | --- |
| `PHI_TRACE` | timestamped startup/transport/session tracing to stderr |
| `PHI_MEDIA_TRACE` | per-node timings for the graph that just ran |
| `PHI_MEDIA_STATE` | what the process still holds at a generation's edges |
| `PHI_VRAM_BUDGET_MB` | override the VRAM ceiling (bounded by the physical card) |
| `PHI_VRAM_MARGIN_PERCENT` | override the budget margin (clamped to 50–100) |
| `PHI_RING_MB` | pin the upload-ring size (to exercise the small-ring paths) |
| `PHI_CUDA_BIN`, `PHI_CUDA_INCLUDE` | extra roots searched for NVRTC DLLs / headers |
| `PHI_TOKENIZER_DIR` | override every tokenizer role at once |
| `PHI_W4A8_CPU` | force the host decode path for grouped-scale weights (A/B) |
| `PHI_PERF` | open the UI with `?perf=1` |
| `PHI_IMG_STEPS`, `PHI_VIDEO_STEPS` | per-run step overrides for the bench |

There are many more, all discovered by `grep getenv core/`; most are
single-purpose experiment switches for kernel tuning.

---

## Tests

The makefile defines host-side test goals that build console programs out of a
handful of `core/` sources, using the same vendored toolchain:

```bash
./make.sh precision-test     # every weight family, per loader, vs a reference
./make.sh sampling-test      # sampler/scheduler maths, no device needed
./make.sh workflow-test      # tool settings → workflow → node plumbing
./make.sh token-test         # context-token estimator vs the real BPE vocab
```

`grouped-decode-test` additionally needs `bin/phi.exe` and a GPU, because it
compares the *device* decode of grouped-scale weights against the host decoder
byte for byte.

> Note: these goals reference sources under `tests/` (and
> `tests/data/*.safetensors`), which are not part of this snapshot — the goals
> are defined, the fixtures are not shipped here. Build them into your tree
> before relying on them.

---

## Security notes

* The UI bridge binds `127.0.0.1` only, and every request is checked against
  the `Host` header, so a page on the web cannot drive the local server.
* `bash` runs whatever the model asks it to. The per-tool permission scope
  (`everywhere` vs `in-cwd`) and the enable switches in Settings → Tool
  settings are the guard rails; set them deliberately.
* API keys are stored in **plain text** in `data/models.json` and
  `data/webui-configs.json`. Treat those files as secrets.
* Sessions record tool arguments and outputs verbatim, including file contents
  the agent read. The `data/sessions/` directory is as sensitive as the
  working directories you point the agent at.

---

## License

This source tree ships no `LICENSE` file. Until one is added, treat the code
as the author's work: no redistribution or commercial use without permission.
Third-party components under `vendor/` (MinGW-w64 GCC, CUDA NVRTC runtime and
headers, the WebView2 loader, ripgrep) keep their own licenses.
