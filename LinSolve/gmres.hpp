#ifndef __LINSOLVE_GMRES_HPP__
#define __LINSOLVE_GMRES_HPP__

#include <chrono>
#include <iostream>
#include <iomanip>

#include "LinSolve/numcepts/numcepts.hpp"
#include "LinSolve/TensorView/TensorView.hpp"

#include "LinSolve/ProgressBar.hpp"
#include "LinSolve/SolverResult.hpp"
#include "LinSolve/linalg.hpp"

namespace linsol
{
  template <typename ViewH, typename ViewC, typename ViewS>
  void givens_rotation(ViewH &&h, ViewC &cs, ViewS &sn, size_t k)
  {
    for (size_t i = 0; i < k; ++i)
    {
      auto h1 = h[i], h2 = h[i + 1];
      h[i] = cs[i] * h1 + sn[i] * h2;
      h[i + 1] = -sn[i] * h1 + cs[i] * h2;
    }

    auto t = linsol::hypot(h[k], h[k + 1]);
    cs[k] = h[k] / t;
    sn[k] = h[k + 1] / t;

    h[k] = cs[k] * h[k] + sn[k] * h[k + 1];
    h[k + 1] = 0.0;
  }

  /**
   * @brief solves leading n x n upper triangular block of the matrix H with
   * right hand side η. Solution overwrites η.
   */
  void triu_solve(size_t n, const tensor::Matrix<double> &H, tensor::Vector<double> &η);
  void triu_solve(size_t n, const tensor::Matrix<float> &H, tensor::Vector<float> &η);
  void triu_solve(size_t n, const tensor::Matrix<std::complex<float>> &H, tensor::Vector<std::complex<float>> &η);
  void triu_solve(size_t n, const tensor::Matrix<std::complex<double>> &H, tensor::Vector<std::complex<double>> &η);

  template <std::floating_point precision>
  struct gmres_options
  {
    size_t restart = 20; // maximum number of iterations before restart.
    size_t maximum_iterations = 100;
    precision maximum_seconds = std::numeric_limits<precision>::infinity();
    precision relative_tolerance = 1e-3;
    precision absolute_tolerance = 1e-6;
    int verbose = 0; // 0 - no output; 1 - progress bar; 2 - one line per iteration.
  };

  /**
   * @brief solves A * x = b using the generalized minimal residual method.
   *
   * @tparam scalar float, double, complex<float>, complex<double>
   * @tparam Operator invocable
   * @param n number of unkowns. Length of x and r.
   * @param x array of length n. On entry, initial guess. On exit, solution.
   * @param A invocable object that computes A * x. Signature: void A(const scalar *x, scalar *y) such that y = A * x.
   * @param b array of length n. The right hand side b.
   * @param options
   * @return SolverResult
   */
  template <numcepts::ScalarType scalar, typename Operator>
  SolverResult gmres(size_t n, scalar *_x, Operator &&A, const scalar *_b, gmres_options<numcepts::precision_t<scalar>> options = {})
  {
    using namespace tensor;
    using real = numcepts::precision_t<scalar>;
    constexpr real one = 1, zero = 0;

    const size_t m = options.restart;
    const size_t m1 = m + 1;

    auto x = reshape(_x, n);
    auto b = reshape(_b, n);

    real bnrm = norm(b);

    Matrix<scalar> V(n, m1);
    Matrix<scalar> H(m1, m);
    Vector<scalar> sn(m);
    Vector<scalar> cs(m);
    Vector<scalar> η(m1);

    auto r = reshape(V.data(), n); // r is the first column of V.

    SolverResult result;
    result.residual_norm.reserve(options.maximum_iterations + 1);
    result.time.reserve(options.maximum_iterations + 1);
    result.num_iter = 0;
    result.num_matvec = 0;
    result.flag = 1;

    A(x.data(), r.data());
    result.num_matvec++;

    for (size_t i = 0; i < n; ++i)
      r(i) = b(i) - r(i);

    real rnrm = norm(r);

    result.residual_norm.push_back((double)rnrm);
    result.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (rnrm < options.relative_tolerance * bnrm + options.absolute_tolerance)
    {
      result.flag = true;

      if (options.verbose)
      {
        std::cout << "After 0 iterations, gmres achieved a relative residual of " << result.residual_norm.back() / bnrm << std::endl;
        std::cout << "gmres successfully converged within desired tolerance." << std::endl;
      }

      return result;
    }

    ProgressBar bar(options.maximum_iterations);
    if (options.verbose)
      std::cout << std::setprecision(5) << std::scientific;

    size_t it = 1;
    for (; it < options.maximum_iterations; ++it)
    {
      scalar β = one / rnrm;
      for (size_t i = 0; i < n; ++i)
        V(i, 0) = β * r(i);

      std::fill(η.begin(), η.end(), (scalar)zero);
      η(0) = rnrm;

      size_t k1 = 0;
      for (size_t k = 0; k < m; ++k)
      {
        k1 = k + 1;
        scalar *vk = V.data() + k * n;
        scalar *vk1 = vk + n;

        A(vk, vk1);
        result.num_matvec++;

        for (size_t j = 0; j < k1; ++j)
        {
          scalar d = dot(V(all(), k1), V(all(), j));
          H(j, k) = d;

          for (size_t i = 0; i < n; ++i)
            V(i, k1) -= d * V(i, j);
        }

        H(k1, k) = norm(V(all(), k1));

        if (H(k1, k) == zero)
          break;

        β = one / H(k1, k);
        for (size_t i = 0; i < n; ++i)
          V(i, k1) *= β;

        givens_rotation(H(all(), k), cs, sn, k);
        η(k1) = -sn(k) * η(k);
        η(k) = cs(k) * η(k);

        if (std::abs(η(k1)) < options.relative_tolerance * bnrm + options.absolute_tolerance)
          break;
      }

      triu_solve(k1, H, η);
      for (size_t k = 0; k < k1; ++k)
      {
        for (size_t i = 0; i < n; ++i)
        {
          x[i] += η(k) * V(i, k);
        }
      }

      A(x.data(), r.data());
      result.num_matvec++;

      for (size_t i = 0; i < n; ++i)
        r(i) = b(i) - r(i);

      rnrm = norm(r);
      result.residual_norm.push_back((double)rnrm);

      auto t1 = std::chrono::high_resolution_clock::now();
      double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      result.time.push_back(dur);

      if (options.verbose == 1)
      {
        ++bar;
        std::cout << bar.get() << " : ||b - A * x|| / ||b|| = " << std::setw(10) << rnrm / bnrm << "\r" << std::flush;
      }
      else if (options.verbose >= 2)
      {
        std::cout << "iteration " << std::setw(10) << it + 1 << " / " << options.maximum_iterations << " : ||b - A * x|| / ||b|| = " << std::setw(10) << rnrm / bnrm << std::endl;
      }

      if (rnrm < options.relative_tolerance * bnrm + options.absolute_tolerance)
      {
        result.flag = 0;
        break;
      }
      if (dur > options.maximum_seconds)
      {
        result.flag = 2;
        if (options.verbose)
          std::cout << "\nMaximum time exceeded for cg. Stopping early." << std::endl;
        break;
      }
    }

    if (options.verbose == 1)
      std::cout << std::endl;
    if (options.verbose)
    {
      std::cout << "After " << it << " iterations, gmres achieved a relative residual of " << rnrm / bnrm << std::endl;
      if (result)
        std::cout << "gmres successfully converged within desired tolerance." << std::endl;
      else
        std::cout << "gmres failed to converge within desired tolerance." << std::endl;
    }

    result.num_iter = it;
    return result;
  }

  /**
   * @brief solves A * x = b using the generalized minimal residual method with preconditioner M.
   *
   * @details Note that this is equivalent to solving M^{-1} A x = M^{-1} b. Therefore, if it is more efficient to combine M and A into a single operator, then do so.
   *
   * @tparam scalar float, double, complex<float>, complex<double>
   * @tparam Operator invocable
   * @tparam Precond invocable
   * @param n number of unkowns. Length of x and r.
   * @param x array of length n. On entry, initial guess. On exit, solution.
   * @param A invocable object that computes A * x. Signature: void A(const scalar *x, scalar *y) such that y = A * x.
   * @param b array of length n. The right hand side b.
   * @param M preconditioner. Invocable object that solves M y == x. Signature: void M(const scalar *x, scalar *y) such that y = M \ x.
   * @param options
   * @return SolverResult
   */
  template <numcepts::ScalarType scalar, typename Operator, typename Precond>
  SolverResult gmres(size_t n, scalar *x, Operator &&A, const scalar *b, Precond &&M, gmres_options<numcepts::precision_t<scalar>> options = {})
  {
    tensor::Vector<scalar> y(n);
    tensor::Vector<scalar> r0(n);

    auto MA = [&](const scalar *u, scalar *v) mutable
    {
      A(u, y.data());
      M(y.data(), v);
    };

    M(b, r0.data());

    return gmres(n, x, MA, r0.data(), options);
  }
} // namespace linsol

#endif