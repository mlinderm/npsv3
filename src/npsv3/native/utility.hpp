#pragma once

#include <stdexcept>
#include <string>

#include <boost/flyweight.hpp>
#include <boost/flyweight/tag.hpp>
#include <boost/flyweight/no_tracking.hpp>

inline void assert_throw(const bool cond, const std::string& text, const std::string& file, const int line) {
  if (!cond) {
    throw std::runtime_error(text + ". In file: " + file + " on line: " + std::to_string(line));
  }
}

#define pyassert(cond, text) assert_throw(cond, text, __FILE__, __LINE__)

namespace npsv3 {
namespace util {

template <class Tag>
class FlyweightStringNoTrack {
  typedef typename boost::flyweight<std::string, boost::flyweights::tag<Tag>, boost::flyweights::no_tracking>
      Flyweight;

 public:
  typedef typename Flyweight::initializer initializer;

  FlyweightStringNoTrack() {}
  FlyweightStringNoTrack(char c) : value_(std::to_string(c)) {}
  FlyweightStringNoTrack(const std::string& s) : value_(s) {}
  FlyweightStringNoTrack(const char* s) : value_(s) {}
  template <typename I>
  FlyweightStringNoTrack(I begin, I end) : value_(begin, end) {}
  FlyweightStringNoTrack(const FlyweightStringNoTrack&) = default;
  FlyweightStringNoTrack(FlyweightStringNoTrack&&) = default;

  operator const std::string&() const { return value_.get(); }
  const std::string& get() const { return value_.get(); }

  /* String Interface */
  typedef std::string::value_type value_type;
  typedef std::string::const_reference const_reference;
  typedef std::string::const_iterator const_iterator;
  typedef std::string::const_reverse_iterator reverse_const_iterator;

  const_iterator begin() const { return this->get().begin(); }
  const_iterator end() const { return this->get().end(); }
  reverse_const_iterator rbegin() const { return this->get().rbegin(); }
  reverse_const_iterator rend() const { return this->get().rend(); }

  size_t size() const { return this->get().size(); }
  bool empty() const { return this->get().empty(); }

  const_reference front() const { return this->get().front(); }
  const_reference back() const { return this->get().back(); }

  const char* c_str() const { return this->get().c_str(); }
  FlyweightStringNoTrack substr(size_t pos = 0, size_t len = std::string::npos) const {
    return FlyweightStringNoTrack(get().substr(pos, len));
  }

  /* Operators */
  FlyweightStringNoTrack& operator=(const FlyweightStringNoTrack& f) {
    value_ = f.value_;
    return *this;
  }
  FlyweightStringNoTrack& operator=(FlyweightStringNoTrack&& f) {
    value_ = f.value_;
    return *this;
  }
  bool operator==(const FlyweightStringNoTrack& f) const { return value_ == f.value_; }
  bool operator!=(const FlyweightStringNoTrack& f) const { return value_ != f.value_; }
  bool operator<(const FlyweightStringNoTrack& f) const { return value_ < f.value_; }

 private:
  Flyweight value_;
};

template <class Tag>
std::ostream& operator<<(std::ostream& ostream, const FlyweightStringNoTrack<Tag>& flyweight) {
  return (ostream << flyweight.get());
}

}  // namespace util
}  // namespace npsv3

namespace std {

template <class T>
struct hash<npsv3::util::FlyweightStringNoTrack<T> > {
  std::size_t operator()(const npsv3::util::FlyweightStringNoTrack<T>& k) const {
    std::hash<const void*> hasher;
    return hasher(&k.get());
  }
};
}  // namespace std