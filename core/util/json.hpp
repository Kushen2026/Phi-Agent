// Minimal ordered JSON value + parser + serializer (UTF-8).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace phi {

class JsonValue {
public:
	enum class Type { Null, Bool, Number, String, Array, Object };
	using Array = std::vector<JsonValue>;
	using Object = std::vector<std::pair<std::string, JsonValue>>;  // insertion-ordered

	JsonValue() = default;
	JsonValue(std::nullptr_t) {}
	JsonValue(bool v) : type_(Type::Bool), bool_(v) {}
	// The integer constructors carry the value exactly as well as in double form
	// (see int_value / as_u64): an id that needs all 64 bits - a media seed is the
	// one that does - would otherwise be rounded to the nearest double before any
	// caller could read it, and the number the caller asked for is the number the
	// run has to be reproducible from.
	JsonValue(int v) : type_(Type::Number), num_((double)v), int_(true), u_((uint64_t)(int64_t)v) {}
	JsonValue(int64_t v) : type_(Type::Number), num_((double)v), int_(true), u_((uint64_t)v) {}
	JsonValue(size_t v)
	    : type_(Type::Number), num_((double)v), int_(true), int_u_((uint64_t)v > (uint64_t)INT64_MAX),
	      u_((uint64_t)v) {}
	JsonValue(double v) : type_(Type::Number), num_(v) {}
	JsonValue(const char* v) : type_(Type::String), str_(v) {}
	JsonValue(std::string v) : type_(Type::String), str_(std::move(v)) {}
	JsonValue(std::string_view v) : type_(Type::String), str_(v) {}

	static JsonValue array() {
		JsonValue v;
		v.type_ = Type::Array;
		v.arr_ = std::make_shared<Array>();
		return v;
	}
	static JsonValue object() {
		JsonValue v;
		v.type_ = Type::Object;
		v.obj_ = std::make_shared<Object>();
		return v;
	}

	Type type() const { return type_; }
	bool int_repr() const { return int_; }  // parsed/constructed as python int
	bool int_unsigned() const { return int_u_; }  // ...and that int is above INT64_MAX
	uint64_t int_value() const { return u_; }     // its exact bits (two's complement if negative)
	bool is_null() const { return type_ == Type::Null; }
	bool is_bool() const { return type_ == Type::Bool; }
	bool is_number() const { return type_ == Type::Number; }
	bool is_string() const { return type_ == Type::String; }
	bool is_array() const { return type_ == Type::Array; }
	bool is_object() const { return type_ == Type::Object; }

	bool as_bool(bool def = false) const { return type_ == Type::Bool ? bool_ : def; }
	double as_number(double def = 0) const { return type_ == Type::Number ? num_ : def; }
	int64_t as_int(int64_t def = 0) const {
		if (type_ != Type::Number) return def;
		// an exact integer needs no double round-trip: only its range is in question
		if (int_) return int_u_ ? def : (int64_t)u_;
		// out-of-range / NaN / inf: the cast is UB. Fall back to the caller's
		// default instead of returning a garbage integer that propagates into
		// array sizes / limits.
		if (!(num_ >= -9.2233720368547758e18 && num_ <= 9.2233720368547758e18)) return def;
		return (int64_t)num_;
	}
	// The UNSIGNED read of the same number, for ids that are u64 end to end. The
	// media tools' `seed` is the case: make_media_seed() draws the whole 64-bit
	// range and a caller that wants a specific run back feeds one of those numbers
	// straight in, so int64_t is simply the wrong width for it (and as_int()
	// answered 0 for the upper half, which the generators read as "draw one").
	uint64_t as_u64(uint64_t def = 0) const {
		if (type_ != Type::Number) return def;
		if (int_) {
			if (!int_u_ && (int64_t)u_ < 0) return def;  // a negative id is no id
			return u_;
		}
		if (!(num_ >= 0.0 && num_ < 18446744073709551616.0)) return def;
		return (uint64_t)num_;
	}
	const std::string& as_string() const {
		static const std::string empty;
		return type_ == Type::String ? str_ : empty;
	}
	// mutable string access (undefined when type != String) for in-place edits
	std::string as_string(const std::string& def) const { return type_ == Type::String ? str_ : def; }
	std::string& str_ref() { return str_; }

	// object access (const: read by key; mutable: get-or-create)
	const JsonValue* find(std::string_view key) const {
		if (type_ != Type::Object) return nullptr;
		// last-wins, matching python json.loads on duplicate keys
		for (auto it = obj_->rbegin(); it != obj_->rend(); ++it) {
			if (it->first == key) return &it->second;
		}
		return nullptr;
	}
	// mutable lookup (no insertion): in-place mutation of stored values without
	// const_cast on internals
	JsonValue* find(std::string_view key) {
		if (type_ != Type::Object) return nullptr;
		for (auto it = obj_->rbegin(); it != obj_->rend(); ++it) {
			if (it->first == key) return &it->second;
		}
		return nullptr;
	}
	JsonValue& operator[](std::string_view key);
	const JsonValue& operator[](std::string_view key) const { return at(key); }
	const JsonValue& at(std::string_view key) const;  // Null when missing

	// array access
	const JsonValue& operator[](size_t i) const {
		static const JsonValue null_v;
		return type_ == Type::Array && i < arr_->size() ? (*arr_)[i] : null_v;
	}
	// mutable element access for in-place edits (undefined when type != Array
	// or i is out of range — callers check size() first)
	JsonValue& operator[](size_t i) { return (*arr_)[i]; }
	void push_back(JsonValue v);
	size_t size() const { return type_ == Type::Array ? arr_->size() : (type_ == Type::Object ? obj_->size() : 0); }
	// empty containers (NOT a deref) when the value has the wrong type: code
	// like j["models"].items() on a missing key used to segfault on the null
	// shared_ptr (fresh install without models.json / a provider entry without
	// models crashed the app at startup)
	const Array& items() const {
		static const Array empty;
		return type_ == Type::Array ? *arr_ : empty;
	}
	const Object& entries() const {
		static const Object empty;
		return type_ == Type::Object ? *obj_ : empty;
	}

	// serialize
	std::string dump(int indent = -1) const;

private:
	// an exact integer payload: `u` holds the value (two's complement when it is
	// negative) and `unsigned_repr` says it is above INT64_MAX, i.e. it has to be
	// rendered unsigned
	static JsonValue make_int(uint64_t u, bool unsigned_repr) {
		JsonValue j;
		j.type_ = Type::Number;
		j.num_ = unsigned_repr ? (double)u : (double)(int64_t)u;
		j.int_ = true;
		j.int_u_ = unsigned_repr;
		j.u_ = u;
		return j;
	}

	Type type_ = Type::Null;
	bool bool_ = false;
	double num_ = 0;
	bool int_ = false;    // render without decimal point
	bool int_u_ = false;  // the exact value is above INT64_MAX (render unsigned)
	uint64_t u_ = 0;      // exact integer value
	std::string str_;
	std::shared_ptr<Array> arr_;    // shared: cheap value copies
	std::shared_ptr<Object> obj_;
	friend struct JsonParser;
};

// Parse; on failure returns std::nullopt and sets *error.
std::optional<JsonValue> json_parse(std::string_view text, std::string* error = nullptr);

}  // namespace phi
