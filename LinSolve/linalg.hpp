#ifndef __LINSOLVE_LINALG_HPP__
#define __LINSOLVE_LINALG_HPP__

#include "LinSolve/numcepts/numcepts.hpp"
#include "LinSolve/TensorView/TensorView.hpp"

namespace linsol
{
  template <numcepts::ScalarType scalar>
  constexpr scalar square(scalar x)
  {
    return x * x;
  }

  template <numcepts::RealType real>
  constexpr real conj(real x)
  {
    return x;
  }

  template <numcepts::RealType real>
  constexpr std::complex<real> conj(std::complex<real> z)
  {
    return std::conj(z);
  }

  template <numcepts::RealType real>
  constexpr real hypot(real a, real b)
  {
    return std::hypot(a, b);
  }

  template <numcepts::ComplexType cx>
  constexpr numcepts::precision_t<cx> hypot(cx a, cx b)
  {
    using real = numcepts::precision_t<cx>;
    real x = square(std::real(a)) + square(std::imag(a)) + square(std::real(b)) + square(std::imag(b));
    return std::sqrt(x);
  }

  template <numcepts::ScalarType scalar>
  numcepts::precision_t<scalar> norm(size_t n, const scalar *x, size_t stride = 1)
  {
    using real = numcepts::precision_t<scalar>;

    real s = 0.0;
    for (size_t i = 0; i < n; i++)
    {
      if constexpr (numcepts::is_complex_v<scalar>)
        s += square(std::real(x[i])) + square(std::imag(x[i]));
      else
        s += square(x[i]);
    }

    return std::sqrt(s);
  }

  template <typename View>
  auto norm(const View &x)
  {
    using scalar = numcepts::value_t<View>;
    using real = numcepts::precision_t<View>;

    real s = 0;
    for (scalar z : x)
    {
      if constexpr (numcepts::is_complex_v<scalar>)
        s += square(std::real(z)) + square(std::imag(z));
      else
        s += square(z);
    }

    return std::sqrt(s);
  }

  template <numcepts::ScalarType scalar>
  scalar dot(size_t n, const scalar *x, const scalar *y, size_t stride_x = 1, size_t stride_y = 1)
  {
    scalar s = 0;
    for (size_t i = 0; i < n; ++i)
      s += linsol::conj(x[i * stride_x]) * y[i * stride_y];
    return s;
  }

  template <typename ViewX, typename ViewY>
  auto dot(const ViewX &x, const ViewY &y)
  {
    using scalar = std::common_type_t<numcepts::value_t<ViewX>, numcepts::value_t<ViewY>>;

    size_t n = x.size();
    if (n != y.size())
      throw std::invalid_argument("Vectors must be of the same size");

    scalar s = 0;
    for (size_t k = 0; k < n; ++k)
      s += linsol::conj(x[k]) * y[k];

    return s;
  }

} // namespace linsol

#endif
