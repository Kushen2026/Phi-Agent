# phi desktop app build (mingw-w64 gcc vendored in vendor/gcc)
# sh-compatible recipes: run from a unix-like shell (git-bash) or use build.bat.

CXX := vendor/gcc/bin/g++.exe
# _WIN32_WINNT/WINVER: the media engine needs the Win10 memory APIs
# (PrefetchVirtualMemory / DiscardVirtualMemory) and the Win10 socket/threading
# declarations.
MEDIA_DEFS := -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00
# vendor/webview2/include is pulled in with -isystem (not -I): it is a vendored
# third-party header that carries MSVC-only `#pragma warning`, which GCC reports
# under -Wunknown-pragmas. Marking the directory as a system include silences
# diagnostics from it without weakening warnings for our own headers.
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -MMD -MP -Icore -Iapp -isystem vendor/webview2/include $(MEDIA_DEFS)
LIBS := -lws2_32 -lwinhttp -lsecur32 -lcrypt32 -lshell32 -lole32 -loleaut32 -lshlwapi -luuid -ladvapi32 -luser32 -lgdi32 -ldwmapi -ldxgi -lversion -static

# Source layout. Flat: every source file is core/<group>/<file>, one level of
# nesting, so a path reads as its concern. Add a directory here and nowhere
# else. The kernel directory is listed apart from the rest only so
# `$(KERNEL_SRC)` can be named as a set.
#
#   core/agent   the assistant loop, sessions, subagents, prompt
#   core/bench   media engine benchmarks
#   core/graph   the node-graph framework (sockets, registry, executor) and its
#                public headers (graph_nodes.hpp / graph_workflows.hpp)
#   core/host    host-side weight decoding: safetensors, quantisation, buffers
#   core/io      image / audio / video / container IO
#   core/kernels CUDA kernels (the only engine code compiled with the GPU path)
#   core/llm     LLM provider clients
#   core/models  the model forward passes (image_dit, av_dit, the VAEs, the
#                text encoders) plus the geometry they share
#   core/net     http client / server
#   core/nodes   one file per node (the catalogue) + its shared service layer
#   core/runtime CUDA runtime, dispatch, scheduling, VRAM budgeting
#   core/sampling samplers and schedulers
#   core/store   settings / session persistence
#   core/text    tokenizer and vision prep
#   core/tools   the agent's tools
#   core/ui      the protocol server and theme
#   core/util    base / json / rtk / utf8 / i18n / the shared media primitives
#   core/workflows one file per ready-made chain
CORE_DIRS := core/util core/net core/llm core/store core/agent core/tools core/ui \
             core/graph core/nodes core/workflows core/runtime core/host core/models \
             core/sampling core/io core/text core/bench
KERNEL_DIR := core/kernels

CORE_SRC := $(foreach d,$(CORE_DIRS),$(wildcard $(d)/*.cpp))
KERNEL_SRC := $(wildcard $(KERNEL_DIR)/*.cpp)
APP_SRC := $(wildcard app/*.cpp)

# Object directory. `make OBJDIR=build/obj/wN` is how an agent isolates its
# objects from a sibling building at the same time: every object is derived from
# OBJDIR (the resource object included), and the paths mirror the source tree,
# so two builds can never share a .o. The default is unchanged, so an ordinary
# `make` still reuses build/obj/.
OBJDIR ?= build/obj

# The -include lines below pull in the .d files written by -MMD. Those are
# dependency rules with no recipe, and the FIRST rule make sees wins the default
# goal
# — so once a single object had been built, a bare `make` silently resolved to
# `build/obj/core/agent/agent.o` and reported "up to date" instead of linking
# phi.exe.
# Pin the goal explicitly.
.DEFAULT_GOAL := all

# Objects mirror the source tree (build/obj/core/io/image_io.o), so two
# files may share a basename without colliding and a stray .o points straight at
# its source.
CORE_OBJ := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CORE_SRC))
KERNEL_OBJ := $(patsubst %.cpp,$(OBJDIR)/%.o,$(KERNEL_SRC))
APP_OBJ := $(patsubst %.cpp,$(OBJDIR)/%.o,$(APP_SRC))
RES_OBJ := $(OBJDIR)/phi_res.o

# -MMD above writes the header list for every object; pull those lists in so
# editing a header rebuilds every translation unit built against it. Without
# this, incremental builds link objects compiled against different struct
# layouts, which corrupts the heap at runtime (c0000374) once the UI connects.
-include $(CORE_OBJ:.o=.d) $(KERNEL_OBJ:.o=.d) $(APP_OBJ:.o=.d)

all: bin/phi.exe

bin/phi.exe: $(CORE_OBJ) $(KERNEL_OBJ) $(APP_OBJ) $(RES_OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -municode -mwindows $(CORE_OBJ) $(KERNEL_OBJ) $(APP_OBJ) $(RES_OBJ) -o bin/phi.exe $(LIBS)
	@# Tolerant like the nvrtc copies below: when phi.exe is already running it
	@# holds bin/webview2loader.dll open, and the copy must not fail the build for
	@# a DLL that is in fact already in place.
	-@cp vendor/webview2/webview2loader.dll bin/ 2>/dev/null || true
	@# NVRTC compiles the CUDA kernels at runtime; nvcuda.dll comes from the
	@# driver. Copy *every* nvrtc runtime the vendored tree ships, so the build is
	@# not tied to one CUDA version: cuda_api.cpp loads the copy whose version is
	@# closest to the installed driver (falling back to a bare-name load from
	@# PATH, or the machine's own CUDA toolkit / pip nvidia-cuda-nvrtc package).
	-@cp vendor/cuda/bin/nvrtc64_*.dll bin/ 2>/dev/null || true
	-@cp vendor/cuda/bin/nvrtc-builtins64_*.dll bin/ 2>/dev/null || true

$(RES_OBJ): app/phi.rc assets/phi.ico
	@mkdir -p $(dir $@)
	@# cd first: windres' preprocessor spawns `vendor` as a command, which
	@# resolves wrongly when invoked from a recipe whose cwd is elsewhere. The
	@# output path is spelled from there too, so OBJDIR applies to the resource
	@# object like it does to every other one.
	cd app && ../vendor/gcc/bin/windres.exe -i phi.rc -o ../$(RES_OBJ) --include-dir ../assets

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Objects mirror the source tree, so this is the whole pattern rule set.

# ── host tests ──────────────────────────────────────────────────────────────
#
# `tests/` holds translation units that are *not* part of phi.exe: they link a
# handful of core sources into a console program and assert a property that a
# full run would only show as "the output looks wrong". Two of them cannot be
# exercised any other way - the precision families have no shipped checkpoint in
# most of the formats, and the sampler/scheduler maths needs no device at all.
#
#   make tests          build and run every one of them
#   make precision-test the goal #1 coverage: every weight family, per loader
#
# They are built with the vendored g++ and the same flags, into build/tests/,
# so `make clean` removes them with everything else.
TESTDIR := build/tests
TEST_CXX := $(CXX) -std=c++20 -O2 -Icore -Itests $(MEDIA_DEFS)
TEST_LIBS := -lversion -lshlwapi -luser32 -static
# The hosts the two test binaries need: the safetensors reader and its decoders,
# the JSON parser the header uses, and the small string/utf8 helpers.
TEST_SUPPORT := core/host/st.cpp core/host/quant.cpp core/host/mtx.cpp \
                core/util/media_common.cpp core/util/json.cpp core/util/base.cpp \
                core/util/utf8.cpp

# Every weight family the engine claims, decoded from a checkpoint written by the
# reference ComfyUI loads these files with (comfy_kitchen) and compared against the
# values that reference's own dequantizers return, plus the routing decision each
# family implies. The fixture and its reference values live in tests/data and
# tests/precision_fixture_ref.hpp (see the test's header for how they were made).
#
# Run from the repo root: the test opens the fixture by its relative path.
precision-test: $(TESTDIR)/precision_coverage_test.exe
	$(TESTDIR)/precision_coverage_test.exe

# The ComfyUI samplers and schedulers: the sigma grids against their contract
# (descending, ending exactly at 0, the right length, `simple` bit-identical to
# the pre-existing implementation) and each integrator against its fixed point.
sampling-test: $(TESTDIR)/sampling_test.exe
	$(TESTDIR)/sampling_test.exe

# The tool settings -> workflow -> node plumbing for the sampler/scheduler pair.
workflow-test: $(TESTDIR)/workflow_test.exe
	$(TESTDIR)/workflow_test.exe

# The context-token estimator against a real BPE vocabulary: the accuracy the
# compaction report and the context meter both rest on. Needs the Qwen2.5
# tokenizer under data/tokenizers/qwen2.5 and skips the accuracy leg without it.
token-test: $(TESTDIR)/token_estimate_test.exe
	$(TESTDIR)/token_estimate_test.exe

# The grouped-scale families' *device* decode against the host decoder it replaces
# (`core/kernels/w4a8_expand.cpp` vs `SafeTensors::dequant_rows`), byte for byte, on
# checkpoint fragments written by the same reference packer the precision test uses.
# Needs bin/phi.exe and a GPU, so it is deliberately not part of `tests`.
# PHI_W4A8_CPU=1 flips the loader to the host path, which is how the A/B is read.
grouped-decode-test: bin/phi.exe
	PHI_BENCH_W4A8=tests/data/w4a8_bench.safetensors bin/phi.exe --media-bench --quick 2>&1 | grep "w4a8 native decode"
	PHI_BENCH_W4A8=tests/data/w6a8_bench.safetensors bin/phi.exe --media-bench --quick 2>&1 | grep "w4a8 native decode"

$(TESTDIR)/precision_coverage_test.exe: tests/precision_coverage_test.cpp                                         tests/precision_fixture_ref.hpp $(TEST_SUPPORT)
	@mkdir -p $(TESTDIR)
	$(TEST_CXX) tests/precision_coverage_test.cpp $(TEST_SUPPORT) -o $@ $(TEST_LIBS)

$(TESTDIR)/sampling_test.exe: tests/sampling_test.cpp core/sampling/samplers.cpp \
                             core/sampling/schedulers.cpp core/sampling/sampling.cpp \
                             core/util/media_common.cpp core/util/base.cpp
	@mkdir -p $(TESTDIR)
	$(TEST_CXX) tests/sampling_test.cpp core/sampling/samplers.cpp \
	    core/sampling/schedulers.cpp core/sampling/sampling.cpp \
	    core/util/media_common.cpp core/util/base.cpp -o $@ $(TEST_LIBS)

$(TESTDIR)/token_estimate_test.exe: tests/token_estimate_test.cpp core/agent/token_estimate.cpp \
                                       core/text/tokenizer.cpp core/text/unicode_cat.cpp \
                                       core/util/base.cpp core/util/json.cpp core/util/utf8.cpp
	@mkdir -p $(TESTDIR)
	$(TEST_CXX) tests/token_estimate_test.cpp core/agent/token_estimate.cpp \
	    core/text/tokenizer.cpp core/text/unicode_cat.cpp core/util/base.cpp \
	    core/util/json.cpp core/util/utf8.cpp -o $@ $(TEST_LIBS)

$(TESTDIR)/workflow_test.exe: tests/workflow_test.cpp core/workflows/image.cpp \
                             core/workflows/video.cpp core/workflows/music.cpp \
                             core/workflows/audio.cpp core/workflows/common.cpp \
                             core/graph/graph.cpp core/util/json.cpp core/util/base.cpp \
                             core/util/media_common.cpp core/util/utf8.cpp
	@mkdir -p $(TESTDIR)
	$(TEST_CXX) tests/workflow_test.cpp core/workflows/image.cpp core/workflows/video.cpp \
	    core/workflows/music.cpp core/workflows/audio.cpp core/workflows/common.cpp \
	    core/graph/graph.cpp core/util/json.cpp core/util/base.cpp core/util/media_common.cpp \
	    core/util/utf8.cpp -o $@ $(TEST_LIBS)

# `precision_coverage_test.cpp`, `token_estimate_test.cpp` and the workflow test
# exist in the tree today; the sampling goal below still has its rule (so
# `make sampling-test` fails with a clear "no rule to make target" rather than
# silently doing nothing) but no source.
tests: precision-test

clean:
	rm -rf build bin

.PHONY: all tests precision-test sampling-test workflow-test token-test grouped-decode-test clean
