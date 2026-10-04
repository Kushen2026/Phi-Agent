# Phi

A local-first Windows desktop AI Agent: it can act as a coding assistant to read and write your codebase, run commands, search the web, and also generate images, video, music, and speech directly on your local GPU.

The entire program is a self-contained Windows executable—no Python dependency, no third-party runtime. GPU computation uses the CUDA driver interface + NVRTC (kernels compiled on demand at runtime), image codec uses WIC, audio/video muxing uses Media Foundation, and the UI is hosted by WebView2.

---

## 1. What It Is

Phi consists of two interlocking parts:

**1. Agent Runtime**

A loop of "think → batch tool calls → observe results → think again." At each step, the model streams reasoning and tool calls, the runtime executes all tools of that step as one batch, appends the results to the conversation history, then proceeds to the next step; when a step produces no tool calls, the text of that step is the final answer.

**2. Local Media Inference Engine**

A multimodal generation engine running on the same GPU, exposing only four tools externally: generate image, generate video, generate music, synthesize speech. Internally it is a set of "node graphs"—loading a model, encoding a prompt, sampling, decoding, writing a file are each a node, connected by typed interfaces. Changing a chain means re-wiring, not rewriting the engine.

---

## 2. Features

### Coding Assistant

- **Read/write code**: precise line-numbered reading (supports large file segmentation), exact string replacement editing, atomic writes, directory listing, regex/literal content search (prefers built-in ripgrep, falls back to native traversal if missing).
- **Execute commands**: runs through a real shell (Git Bash preferred, falls back to cmd.exe), supports pipes, redirection, &&; output streams back to the UI in real time and is automatically compressed—strips ANSI escapes, collapses build/git noise, keeps head and tail of overly long output.
- **Network access**: Bing search (returns numbered result list: title / URL / snippet) and web page fetching (HTML to plain text).
- **Multimodal reading**: images are passed directly to the model as image content; videos can specify frame numbers, extracted frames are likewise passed as images. Both are first scaled to a size the model can resolve and re-encoded, avoiding a screenshot repeatedly inflating in every request round.

### Background Subagent

The main model can hand off an entire block of tasks to a brand-new subagent: it has an independent context window, an independent toolset (cannot spawn further subagents within a subagent), and the same system prompt as the main session, advancing in parallel on its own thread. The subagent's run log streams back in real time in the session's own message format to the corresponding tool card; upon task completion, its last text segment is returned to the main model as a report.

### Local Media Generation

Four tools, all running on the local GPU, no network, no external services required:

- **Image generation**: text-to-image, or editing/inpainting with a reference image. Output dimensions are determined solely by width/height; the reference image is scaled only by its own resolution and does not participate in determining the canvas.
- **Video generation**: joint audio-video generation, supports reference image / reference video (including its audio track) / independent reference audio, outputs H.264 + AAC mp4.
- **Music generation**: given style tags, lyrics, and music metadata (BPM, duration, time signature, key, language), generates 48 kHz music.
- **Speech synthesis**: synthesizes speech using "text to speak + natural language instruction describing the voice timbre." Only voice design, no voice cloning—the engine does not accept reference audio.

### Sessions and UI

- Sessions are persisted as append-only JSONL, can be reopened, continued, deleted (deletion goes to recycle bin) at any time.
- Model, thinking intensity, working directory, system prompt language can all be switched at runtime.
- When context is too long, manual compression is possible: summarize history into a summary, after which all requests carry only the summary and subsequent content.

---

## 3. Architecture

### Layers

    Desktop Shell (Native Window + WebView2)
      Placeholder splash / theme / folder picker
              │  Loopback HTTP + WebSocket
              ▼
    Protocol Layer: Session Facade + Snapshot Projection
      Projects Agent state into JSON the UI can render directly
              │
              ▼
    Agent Loop
      THINK: Streaming model response
      EXECUTE: Batch execute all tools for this step
        │                    │
        ▼                    ▼
    Tool Layer            Media Engine
    bash/read/...         Node Graph + CUDA
    web/subagent          VRAM Ledger

### Agent Loop

- **A turn** = one user input driven to completion.
- **A step** = one streaming model call + one batch of all tools in this step.
- The loop alternates steps until a step produces no tool calls.

A few deliberate design points:

- Once tool calls appear in the response, prose streamed in the same step is discarded—only the text of the final summary step is retained. "Silently do tool steps, only speak at the end" is enforced by the loop itself, not requested via prompt.
- Tool results are written to history **at the moment each call is answered**, not after the whole batch finishes. This way, even if a tool's UI callback throws an exception, or the user aborts midway, history will not contain a dangling state of "tool_call without corresponding tool_result"—such history would cause every subsequent request to be rejected by the upstream.
- On stream interruption (network error, HTTP 4xx), already recorded tool calls are still filled in one by one with placeholder results, for the same reason as above.
- For parallel tool calls within the same turn, their IDs are forcibly deduplicated and completed: some gateways stuff multiple parallel calls with the same placeholder ID, and both wire formats pair by ID; duplicate IDs would make the whole batch appear "under-answered."
- Users can "insert" new input during streaming (steering), which will be merged into context before the next step begins.

### Tool Layer

Tools fall into four categories:

| Category | Description |
| --- | --- |
| File/Command | Read/write, search, execute commands within the session working directory |
| Network | Search, fetch web pages |
| Subagent | Hand off task to background subagent |
| Media | Image / video / music / speech generation |

Unified conventions:

- **Every built-in prompt string has both Chinese and English versions**, determined by the session language (settings.json's systemPromptLanguage), including parameter descriptions, error messages, and explanatory lines inserted in compressed output. No place hardcodes language.
- **Parameter schemas are generated on the fly at each request according to the current language**, so switching language does not require a restart.
- **Permission gate**: each tool can be set to "within working directory only"; out-of-bounds access is rejected with a readable reason.
- **Output cap**: each tool can be individually configured with a maximum output character count.

### Media Engine

The engine is process-level resident, lazily loaded on demand: the image backbone's weights plus text encoder add up to over a dozen GB; reopening each call would dominate the entire runtime, while opening at startup would make sessions that never use media pay an unnecessary cost.

The full flow of a chain is **one node graph**:

    Model Load ─┬─ Prompt Encode ─┐
    Text Load ──┘                 ├─ Sampler ─ Decode ─ Write File
    Empty Latent ─────────────────┘
    Scheduler (sigma grid) ───────┘

- Nodes have **typed interfaces** (MODEL / CLIP / VAE / LORA / CONDITIONING / LATENT / SIGMAS / IMAGE / AUDIO / VIDEO / scalar). The executor traverses in dependency order; type mismatches are rejected before the node runs.
- Errors are **values, not exceptions**: if a node cannot run, it states which type it is, which id, and why.
- **A node corresponds to a stage, not a model**. Samplers dispatch by backbone type, VAE decoding dispatches by the VAE's role, latent nodes dispatch by their own geometry parameters. Adding a new backbone is adding a branch, not adding a set of nodes.
- **Precision is determined by the file**. The loader reads the precision declared by the checkpoint itself (int8, 4/6 bit packed, fp8, fp16, bf16, fp32), loads at that precision and computes at that precision, without doing a "convert everything to one type" conversion. The same model exported at different precisions is just a different filename in settings.
- **Weights are stream-loaded in blocks**. When the GPU cannot hold the entire checkpoint, only the current block is pushed into VRAM, then the next block after computation.
- **Runtime LoRA correction**. For streamed weights that need to be re-read repeatedly, LoRA is not folded into the weights (that would mean re-decoding, adapting, and re-quantizing the entire checkpoint every step), but acts as a post-term of GEMM: y = x·W + (x·A)·B. Geometry is completely equivalent, at the cost of a few percentage points of the base GEMM, and precision is actually higher (delta kept in fp32).

### VRAM Management

This is the most constrained resource in the entire engine, so there is a unified ledger:

- Every byte of resident VRAM is accounted for before allocation; exceeding the limit is **rejected** with a readable error, never silently shrunk.
- The limit is not a constant, but **derived per card**: how much the driver is willing to give this process, and how much the card itself has, take the smaller; re-read before each tool call, so "a card that was occupied by another program a minute ago" is now planned truthfully.
- **Resident window**: as many blocks as VRAM can hold stay resident, the rest are stream-loaded. Dynamically increased or decreased at runtime based on measured usage, with the goal of keeping the sampling phase in the 90–100% range of the driver limit—below this range is paying disk bandwidth for nothing, above it is on the edge of losing the device.
- **Return at stage boundaries**: at the end of each generation stage, VRAM not belonging to the next stage is returned to the device. Otherwise, a finished video task would think "3 GB is still occupied on the card" and cause the immediately following image task to keep several fewer blocks resident.

### Media Tools Also Use Node Graphs

The four media tools are the same executor, the same loader, the same nodes, just different wiring. So "adding a new chain" is writing a new wiring, not writing a new set of modules—the speech synthesis chain reuses the existing loader and audio-writing node without changing a single line.

---

## 4. Usage

### Running

Double-click the executable directly. The window appears on screen within milliseconds (with a theme-colored startup placeholder), the browser kernel and protocol service start in parallel in the background, and hand off to the UI once the first frame is ready.

Command-line arguments:

    phi.exe --media-bench [options]       Media engine acceptance benchmark
    phi.exe --media-tool <name> [...]     Run a media tool directly (bypassing the model)

Subcommands of --media-tool:

- name schema —— Print the tool's parameter schema (English).
- name graph —— Only build and validate the node graph; no GPU, checkpoint, or model needed.
- name resolve —— Print which file each role resolves to under current settings, and whether it exists.
- name '{...}' —— Actually run once with a JSON parameter, returning the result text.

### First-Time Configuration

**1. Configure Model**

In the UI, fill in at least one API configuration: provider, base URL, API key, model. Two wire formats are supported:

- anthropic-messages
- openai-completions (including various compatible gateways)

Configurations are saved, multiple can be kept and the "currently effective" one switched. The model list and thinking intensity options are filtered by the model's own declared capabilities.

**2. Choose Working Directory**

All relative paths of the Agent are resolved relative to the session working directory. The working directory can be switched in the UI (which rebuilds the session), or written into settings before startup.

**3. Place Media Models (Optional)**

If you want to use media generation capabilities, put the corresponding checkpoints into the models directory. The directory layout is free:

- Subdirectories per chain (one directory per chain, weights, vocab, config each in place), or
- All flat under the models directory.

The settings panel lists **all** files under the models directory, and for each role (backbone weights, text encoder, VAE, vocab, merge table, tokenizer config, LoRA, etc.) you select one. That is: **the filesystem is the model directory**—put a file in, it appears; no need to maintain a built-in list, and no need to remember some format marker; the loader detects precision itself.

Each chain's selections are independent of each other. LoRA is also a separate list per chain, stacked in order; image LoRA will never leak into the video chain.

**4. Tool Settings**

- Tool enable/disable switches.
- Per-tool permissions (e.g., restricted to within working directory).
- Per-tool output cap.
- Tool subset available to subagents.
- Sampling steps, sampler, scheduler name for each chain (leave empty to use that chain's own release defaults).

### Daily Use

- **Conversation**: just ask. The Agent will decide on its own whether to read files, run commands, search, or generate media.
- **Abort midway**: stop at any time. The running stream is disconnected, the running command is killed (along with its entire process tree), the running subagent receives a cancellation signal.
- **Insert supplement**: continue typing during generation; new content is merged into context before the next thinking step.
- **Compress context**: when the conversation gets long, manually trigger compression; after compression all requests carry only the summary and subsequent content.
- **Session management**: new, switch, rename, delete (to recycle bin). Empty sessions with no messages written are automatically discarded and do not accumulate on disk.
- **Switch model / switch thinking intensity**: takes effect immediately and is written to the current session.

### Environment Requirements

- Windows 10 or higher.
- WebView2 Runtime (preinstalled on most Windows 10/11).
- Using media generation capabilities requires an NVIDIA GPU supporting CUDA; when CUDA is missing, media tools give a readable error, and other features are unaffected.
- For media generation, it is recommended to place checkpoints in a models directory on a non-system drive—the engine stream-loads weights in blocks, and disk throughput directly affects the time per step.

---

## 5. Design Trade-offs

- **No Python, no third-party runtime**. The entire inference engine uses only the system's built-in codec components and the CUDA driver interface. CUDA kernels are compiled at runtime with NVRTC (so no MSVC toolchain is needed), and compilation results are cached to disk.
- **Disk is slow, so weights should not be read repeatedly**. Checkpoints are usually larger than memory, so the engine keeps "what can be resident resident, the rest stream-loaded," and caches repeatedly used weight blocks in host memory.
- **VRAM is scarce, so the ledger must be honest**. All planning starts from "how much this machine can really give right now," not from a hardcoded constant. This is why the same binary can run on a 6 GB laptop GPU and a 48 GB workstation.
- **Precision is decided by the file**. The engine does not decide for the user "which precision should be used"—as many exports as a model has, that many it can read; changing precision is changing a filename.
- **Language is a session property**. All built-in strings (tool descriptions, parameter descriptions, error messages, explanatory lines in compressed output) follow the session language and are not hardcoded in code.
- **Failures must be readable**. Almost every error message carries specific numbers: how much is needed, how much is missing, what the limit is, which file, which node. Rejecting a run is acceptable; silently producing wrong results is not.