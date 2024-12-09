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
static int pcg_test(numcepts::precision_t<scalar> rtol, numcepts::precision_t<scalar> atol)
{
  using real = numcepts::precision_t<scalar>;

  size_t n = 100;

  auto A = [n](const scalar *x, scalar *y)
  {
    scalar two(2);
    y[0] = two * x[0] - x[1];
    for (size_t i = 1; i + 1 < n; ++i)
      y[i] = -x[i - 1] + two * x[i] - x[i + 1];
    y[n - 1] = -x[n - 2] + two * x[n - 1];
  };

  auto M = [n](const scalar *x, scalar *y)
  {
    scalar a = M_SQRT1_2;
    y[0] = a * x[0];
    for (size_t i = 1; i < n; ++i)
      y[i] = a * (x[i] + a * y[i - 1]);

    y[n - 1] *= a;
    for (size_t i = 1; i < n; ++i)
      y[n - 1 - i] = a * (y[n - 1 - i] + a * y[n - i]);
  };

  Vector<scalar> x(n);
  Vector<scalar> b(n);
  Vector<scalar> y(n);

  for (auto &u : y)
    u = randu<scalar>();

  A(y, b);

  auto result = cg(n, x.data(), A, b.data(), M, {.maximum_iterations = 100, .relative_tolerance = rtol, .absolute_tolerance = atol, .verbose = 1});

  for (size_t i = 0; i < n; ++i)
    x(i) -= y(i);

  real error = norm(x);
  real bnrm = norm(b);
  real q = 4.9e2; // ||e|| <= q * √(r, M\r)

  real tol = q * (bnrm * rtol + atol);
  if (error > tol)
  {
    std::cout << "gmres_test<" << typeid(scalar).name() << "> failed because the error = " << error << " > " << tol << " (the expected tolerance).\n";
    return 1;
  }

  return 0;
}

template <numcepts::ScalarType scalar>
static int cg_test(numcepts::precision_t<scalar> rtol, numcepts::precision_t<scalar> atol)
{
  using real = numcepts::precision_t<scalar>;

  size_t n = 100;

  auto A = [n](const scalar *x, scalar *y)
  {
    scalar two(2);
    y[0] = two * x[0] - x[1];
    for (size_t i = 1; i + 1 < n; ++i)
      y[i] = -x[i - 1] + two * x[i] - x[i + 1];
    y[n - 1] = -x[n - 2] + two * x[n - 1];
  };

  Vector<scalar> x(n);
  Vector<scalar> b(n);
  Vector<scalar> y(n);

  for (auto &u : y)
    u = randu<scalar>();

  A(y, b);

  auto result = cg(n, x.data(), A, b.data(), {.maximum_iterations = 100, .relative_tolerance = rtol, .absolute_tolerance = atol, .verbose = 1});

  for (size_t i = 0; i < n; ++i)
    x(i) -= y(i);

  real error = norm(x);
  real bnrm = norm(b);
  real q = 1.1e3; // ||e|| <= q * ||r||

  real tol = q * (bnrm * rtol + atol);
  if (error > tol)
  {
    std::cout << "gmres_test<" << typeid(scalar).name() << "> failed because the error = " << error << " > " << tol << " (the expected tolerance).\n";
    return 1;
  }

  return 0;
}

int main()
{
  int fails = 0;

  fails += pcg_test<float>(1e-5, 1e-7);
  fails += pcg_test<double>(1e-7, 1e-10);
  fails += pcg_test<std::complex<float>>(1e-5, 1e-7);
  fails += pcg_test<std::complex<double>>(1e-7, 1e-10);

  fails += cg_test<float>(1e-5, 1e-7);
  fails += cg_test<double>(1e-7, 1e-10);
  fails += cg_test<std::complex<float>>(1e-5, 1e-7);
  fails += cg_test<std::complex<double>>(1e-7, 1e-10);

  return fails;
}