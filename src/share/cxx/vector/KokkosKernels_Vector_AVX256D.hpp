#ifndef __KOKKOSKERNELS_VECTOR_AVX256D_HPP__
#define __KOKKOSKERNELS_VECTOR_AVX256D_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#if defined(__AVX__) || defined(__AVX2__)

#include <immintrin.h>
#include <assert.h>

namespace KokkosKernels {
namespace Batched {
namespace Experimental {

///
/// AVX256D double
///

template <typename SpT> class Vector<VectorTag<AVX<double, SpT>, 4> > {
public:
  using type = Vector<VectorTag<AVX<double, SpT>, 4> >;
  using value_type = double;
  using real_type = double;

  enum : int { vector_length = 4 };

  union data_type {
    __m256d v;
    double d[4];
  };

  KOKKOS_INLINE_FUNCTION
  static const char *label() { return "AVX256"; }

private:
  mutable data_type _data;

public:
  inline Vector() { _data.v = _mm256_setzero_pd(); }
  inline Vector(const value_type val) { _data.v = _mm256_set1_pd(val); }
  inline Vector(const type &b) { _data.v = b._data.v; }
  inline Vector(__m256d const &val) { _data.v = val; }

  inline type &operator=(__m256d const &val) {
    _data.v = val;
    return *this;
  }

  inline operator __m256d() const { return _data.v; }

  inline type &loadAligned(value_type const *p) {
    _data.v = _mm256_load_pd(p);
    return *this;
  }

  inline type &loadUnaligned(value_type const *p) {
    _data.v = _mm256_loadu_pd(p);
    return *this;
  }

  inline void storeAligned(value_type *p) const { _mm256_store_pd(p, _data.v); }

  inline void storeUnaligned(value_type *p) const {
    _mm256_storeu_pd(p, _data.v);
  }

#ifdef NDEBUG
  // Does nothing in non-debug mode
  KOKKOS_INLINE_FUNCTION
  void debug_set_invalid(int left, int right) {}
#else
  // left, right specify the closed range of indices to set to quiet NaNs
  KOKKOS_INLINE_FUNCTION
  void debug_set_invalid(int left, int right) {
    for(int i = left; i <= right; i++) {
      _data.d[i] = 0.0 / 0.0;
    }
  }
#endif // NDEBUG

  KOKKOS_INLINE_FUNCTION
  void shift_left(int num_values) {
    assert(num_values > 0);
    for(int i = 0; i < vector_length - num_values; i++) {
      _data.d[i] = _data.d[i + num_values];
    }
    debug_set_invalid(vector_length - num_values, vector_length - 1);
  }

  KOKKOS_INLINE_FUNCTION
  void shift_right(int num_values) {
    assert(num_values > 0);
    for(int i = vector_length - 1; i >= num_values; i--) {
      _data.d[i] = _data.d[i - num_values];
    }
    debug_set_invalid(0, num_values - 1);
  }

  inline value_type &operator[](int i) const { return _data.d[i]; }
};

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator+(Vector<VectorTag<AVX<double, SpT>, 4> > const &a,
          Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return _mm256_add_pd(a, b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator+(Vector<VectorTag<AVX<double, SpT>, 4> > const &a, const double b) {
  return a + Vector<VectorTag<AVX<double, SpT>, 4> >(b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator+(const double a, Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return Vector<VectorTag<AVX<double, SpT>, 4> >(a) + b;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator+=(Vector<VectorTag<AVX<double, SpT>, 4> > &a,
           Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  a = a + b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator+=(Vector<VectorTag<AVX<double, SpT>, 4> > &a, const double b) {
  a = a + b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator++(Vector<VectorTag<AVX<double, SpT>, 4> > &a, int) {
  Vector<VectorTag<AVX<double, SpT>, 4> > a0 = a;
  a = a + 1.0;
  return a0;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator++(Vector<VectorTag<AVX<double, SpT>, 4> > &a) {
  a = a + 1.0;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator-(Vector<VectorTag<AVX<double, SpT>, 4> > const &a,
          Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return _mm256_sub_pd(a, b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator-(Vector<VectorTag<AVX<double, SpT>, 4> > const &a, const double b) {
  return a - Vector<VectorTag<AVX<double, SpT>, 4> >(b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator-(const double a, Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return Vector<VectorTag<AVX<double, SpT>, 4> >(a) - b;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator-=(Vector<VectorTag<AVX<double, SpT>, 4> > &a,
           Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  a = a - b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator-=(Vector<VectorTag<AVX<double, SpT>, 4> > &a, const double b) {
  a = a - b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator--(Vector<VectorTag<AVX<double, SpT>, 4> > &a, int) {
  Vector<VectorTag<AVX<double, SpT>, 4> > a0 = a;
  a = a - 1.0;
  return a0;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator--(Vector<VectorTag<AVX<double, SpT>, 4> > &a) {
  a = a - 1.0;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator*(Vector<VectorTag<AVX<double, SpT>, 4> > const &a,
          Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return _mm256_mul_pd(a, b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator*(Vector<VectorTag<AVX<double, SpT>, 4> > const &a, const double b) {
  return a * Vector<VectorTag<AVX<double, SpT>, 4> >(b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator*(const double a, Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return Vector<VectorTag<AVX<double, SpT>, 4> >(a) * b;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator*=(Vector<VectorTag<AVX<double, SpT>, 4> > &a,
           Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  a = a * b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator*=(Vector<VectorTag<AVX<double, SpT>, 4> > &a, const double b) {
  a = a * b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator/(Vector<VectorTag<AVX<double, SpT>, 4> > const &a,
          Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return _mm256_div_pd(a, b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator/(Vector<VectorTag<AVX<double, SpT>, 4> > const &a, const double b) {
  return a / Vector<VectorTag<AVX<double, SpT>, 4> >(b);
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator/(const double a, Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  return Vector<VectorTag<AVX<double, SpT>, 4> >(a) / b;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator/=(Vector<VectorTag<AVX<double, SpT>, 4> > &a,
           Vector<VectorTag<AVX<double, SpT>, 4> > const &b) {
  a = a / b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> > &
operator/=(Vector<VectorTag<AVX<double, SpT>, 4> > &a, const double b) {
  a = a / b;
  return a;
}

template <typename SpT>
inline static Vector<VectorTag<AVX<double, SpT>, 4> >
operator-(Vector<VectorTag<AVX<double, SpT>, 4> > const &a) {
  return -1 * a;
}

} // Experimental
} // Batched
} // KokkosKernels

#endif
#endif
