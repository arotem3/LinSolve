#include "linsolve.hpp"

using namespace tensor;
using namespace linsol;

template <numcepts::ScalarType scalar>
static scalar randu()
{
  if constexpr (numcepts::is_complex_v<scalar>)
  {
    using real = numcepts::precision_t<scalar>;
    scalar i(0, 1);
    real u = (real)rand() / RAND_MAX;
    real v = (real)rand() / RAND_MAX;
    real r = std::sqrt(u);
    real θ = 2.0 * M_PI * v;
    return r * std::exp(i * θ);
  }
  else
  {
    scalar u = (scalar)rand() / RAND_MAX;
    return 2 * u - 1;
  }
}

template <numcepts::ScalarType scalar>
static int gmres_test(numcepts::precision_t<scalar> rtol, numcepts::precision_t<scalar> atol)
{
  using real = numcepts::precision_t<scalar>;

  size_t n = 100;

  auto A = [n](const scalar *x, scalar *y)
  {
    y[0] = x[1];
    for (size_t i = 1; i + 1 < n; ++i)
      y[i] = x[i - 1] + x[i + 1];
    y[n - 1] = x[n - 2];
  };

  auto M = [n](const scalar *x, scalar *y)
  {
    std::copy_n(x, n, y); // identity
  };

  Vector<scalar> x(n);
  Vector<scalar> b(n);
  Vector<scalar> y(n);

  for (auto &u : y)
    u = randu<scalar>();

  A(y, b);
  real bnrm = BLAS::norm(n, b.data(), 1);

  auto result = gmres(n, x.data(), A, b.data(), M, {.maximum_iterations = 1000, .relative_tolerance = rtol, .absolute_tolerance = atol, .verbose = 1});

  for (size_t i = 0; i < n; ++i)
    x(i) -= y(i);

  real error = BLAS::norm(n, x.data(), 1);
  real invA = 33.0; // upper estimate of the norm of inv(A).

  // ||e|| = ||inv(A)*r|| <= ||inv(A)||*||r|| <= ||inv(A)||*(||b||*rtol + atol)
  if (error > invA * (bnrm * rtol + atol))
  {
    std::cout << "gmres_test<" << typeid(scalar).name() << "> failed because the error = " << error << " > " << invA * (bnrm * rtol + atol) << " (the expected tolerance).\n";
    return 1;
  }

  return 0;
}

int main()
{
  int fails = 0;

  fails += gmres_test<float>(1e-5, 1e-7);
  fails += gmres_test<double>(1e-8, 1e-10);
  fails += gmres_test<std::complex<float>>(1e-5, 1e-7);
  fails += gmres_test<std::complex<double>>(1e-8, 1e-10);

  return fails;
}