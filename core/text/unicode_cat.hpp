// Unicode classification for the tokeniser's pre-tokenisation step.
//
// The tables live in unicode_cat.cpp, generated from the Unicode Character
// from the Unicode Character Database). Only the three classes the Qwen2
// pre-tokeniser regex asks about are provided: it never needs general
// categories beyond these, and a full category table would be far larger for no
// gain.
#pragma once

namespace phi::media {

bool unicode_is_letter(unsigned cp);   // \p{L}
bool unicode_is_number(unsigned cp);   // \p{N}
bool unicode_is_space(unsigned cp);    // \s  (Python/Rust definition)

struct cp_range {
	unsigned lo;
	unsigned hi;  // inclusive
};

// Binary search over an ascending, non-overlapping range table.
inline bool cp_in_ranges(const cp_range* ranges, unsigned count, unsigned cp) {
	unsigned lo = 0, hi = count;
	while (lo < hi) {
		unsigned mid = lo + (hi - lo) / 2;
		if (cp < ranges[mid].lo) {
			hi = mid;
		} else if (cp > ranges[mid].hi) {
			lo = mid + 1;
		} else {
			return true;
		}
	}
	return false;
}

}  // namespace phi::media
