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

  class BLAS
  {
  public:
    // returns x^H * y
    template <numcepts::ScalarType scalar>
    static scalar dot(int n, const scalar *x, int incx, const scalar *y, int incy)
    {
      scalar s = 0;
      for (int i = 0; i < n; ++i)
        s += linsol::conj(x[i * incx]) * y[i * incy];
      return s;
    }

    // returns ||x||_2
    template <numcepts::ScalarType scalar>
    static numcepts::precision_t<scalar> norm(int n, const scalar *x, int incx)
    {
      using real = numcepts::precision_t<scalar>;

      real s = 0.0;
      for (int i = 0; i < n; i++)
      {
        if constexpr (numcepts::is_complex_v<scalar>)
          s += square(std::real(x[i * incx])) + square(std::imag(x[i * incx]));
        else
          s += square(x[i * incx]);
      }

      return std::sqrt(s);
    }

    // y <- alpha * x + y
    template <numcepts::ScalarType scalar>
    static void axpy(int n, scalar alpha, const scalar *x, int incx, scalar *y, int incy)
    {
      for (int i = 0; i < n; ++i)
        y[i * incy] += alpha * x[i * incx];
    }

    // x <- alpha * x
    template <numcepts::ScalarType scalar>
    static void scal(int n, scalar alpha, scalar *x, int incx)
    {
      for (int i = 0; i < n; ++i)
        x[i * incx] *= alpha;
    }

    // y <- x
    template <numcepts::ScalarType scalar>
    static void copy(int n, const scalar *x, int incx, scalar *y, int incy)
    {
      for (int i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
    }
  };

} // namespace linsol

#endif
