// Script-aware token estimation for the agent's context accounting.
//
// The context meter has to answer one question: how big is the next request?
// The provider reports its own count after a call, but between calls — while a
// turn streams, and always for a compaction summary the provider never bills —
// nothing authoritative exists. This file is that fallback.
//
// Why not "four characters per token":
//
// The previous estimate charged one token per non-ASCII code point and
// `bytes / 4` for the ASCII remainder. Measured against the Qwen2.5 byte-level
// BPE vocabulary that ships with the image encoders (data/tokenizers/qwen2.5),
// that is wrong in both directions at once:
//
//   * Chinese prose costs ~0.66 token per ideograph, not 1.0 — common words and
//     punctuation merge — so pure CJK history was over-priced ~40 %.
//   * Source code, build logs and serialized JSON cost 0.30–0.50 token per
//     *byte* (2–3 bytes per token), not 0.25 — brackets, commas and identifiers
//     are rarely in the merge table — so tool output was under-priced by up to
//     60 %.
//
// A meter that is 40 % high in one place and 60 % low in another is worse than
// a merely coarse one: the context bar jumps when the conversation switches
// from prose to tool output, and a compaction report sinks for exactly the
// Chinese sessions that need it most.
//
// The model below therefore classifies every code point into the structural
// classes a byte-level BPE actually distinguishes, and prices each class at the
// density measured for it. Letters are handled differently from everything
// else: a BPE learns frequent whole *words* as single tokens, so one common
// English word — however long — is often exactly one token, while a rare or
// made-up word falls apart into sub-words. Pricing a word as "one token plus a
// small charge for the letters past the sixth, and a larger one past the tenth"
// reproduces both ends of that curve, and is what keeps long identifiers and
// rare vocabulary from being under-counted while prose is not over-counted.
//
// Fitting the nine coefficients against 27 samples — C++, JavaScript, HTML,
// CSS, JSON, a makefile, Phi's own generated tool schemas, and English and
// Chinese prose — puts this estimator within ~4 % RMS and ~10 % worst-case of
// the real vocabulary, versus 57 % worst-case before. Leave-one-out refits stay
// inside the same band, so the coefficients describe the vocabulary rather than
// the sample.
//
// Known limits, so nobody mistakes this for a tokenizer:
//
//   * Densely packed JSON with no spaces (`{"a":1,"b":2,...}`) is under-priced
//     by ~35 %: with no spaces to break them up, key/value runs merge into
//     patterns this model prices as ordinary punctuation. Tool schemas as Phi
//     actually serializes them (with spaces and newlines) land within ~5 %.
//   * Base64 and other high-entropy alphabets are under-priced by ~40 % for the
//     same reason. Pictures are never priced this way, though: they are booked
//     at the flat visual price below.
//   * Both limits are one-sided (low), which is the direction a context meter
//     is allowed to err in only because provider usage, when it exists, wins.
//
// Two properties are deliberate:
//
//   * It is a pure function of the bytes — no tokenizer call. Loading the 151k
//     entry vocabulary just to size a request would cost megabytes and a
//     dictionary scan inside the snapshot path, which runs on every streaming
//     tick. The estimator is called there, so it stays O(bytes) and stateless.
//   * It never returns a negative or shrinking count: every term is
//     non-negative, and the framing overheads are added on top rather than
//     folded into the text density.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "agent/agent.hpp"

namespace phi {

namespace tokens {

// ── densities ───────────────────────────────────────────────────────────────
//
// Tokens per unit, calibrated against the Qwen2.5 vocabulary (see the file
// header). They are the same order of magnitude for every BPE family the
// providers here use — all are trained on the same mixture of English prose,
// source code and CJK — so one table serves both Anthropic's and the
// OpenAI-compatible endpoints. Where the families really differ, the provider's
// own reported usage wins; see Agent::estimate_context_tokens and
// AgentSessionHost::compact.
inline constexpr double kWordRank = 0.904;       // per ASCII "word" (a run of letters)
inline constexpr double kExtraLetterRank = 0.020; // per letter past the 6th in a word
inline constexpr double kLongLetterRank = 0.100;  // per letter past the 10th in a word
inline constexpr double kDigitRank = 1.15;        // per ASCII digit: BPE splits digit runs per char
inline constexpr double kSpaceRank = 0.185;       // per ASCII space/tab/newline
inline constexpr double kPunctRank = 0.571;       // per ASCII punctuation mark
inline constexpr double kCjkRank = 0.660;         // per Han/kana/hangul ideograph
inline constexpr double kCjkPunctRank = 0.757;    // per full-width punctuation mark
inline constexpr double kOtherRank = 0.800;       // per other non-ASCII code point (emoji, symbols)

// Structural framing the providers pay per element and that no text density
// sees: the role and separator fields of a chat message, and the JSON shell
// around each typed content block.
inline constexpr int64_t kMessageOverhead = 4;
inline constexpr int64_t kBlockOverhead = 4;

// An image is billed by the provider's own vision pricing, not by its base64
// text. Phi's picture path resamples anything past 1568 px on the long side
// (tools_helpers.cpp, kModelImageMaxSide), and the endpoints here charge a
// flat visual price per picture in the 1k–1.6k range, so one picture is booked
// as this many tokens instead of as megabytes of base64.
inline constexpr int64_t kImageTokens = 1200;

// A whole video handed to the model is billed per sampled frame.
inline constexpr int64_t kVideoTokens = 9000;

// The calibrated density of `text` alone, in tokens (no framing overhead).
int64_t estimate_text(std::string_view text);

// One model-visible message, including its role framing and the per-block
// overhead of every content part. Images and videos are booked at the flat
// visual prices above, never at the size of their base64 payload.
int64_t estimate_message(const AgentMessage& message);

// The tool-schema array as the request sends it (Agent::build_tools_json).
int64_t estimate_tools_json(std::string_view tools_json);

// The whole request: system prompt + tool schemas + every message the model
// will see (already run through the compaction transform by the caller).
// This is the number the context meter and the compaction report should both
// be reading.
int64_t estimate_request(std::string_view system_prompt, std::string_view tools_json,
	const std::vector<AgentMessage>& messages);

}  // namespace tokens
}  // namespace phi
