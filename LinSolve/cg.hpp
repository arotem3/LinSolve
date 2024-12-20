#ifndef __LINSOLVE_CG_HPP__
#define __LINSOLVE_CG_HPP__

#include "LinSolve/ProgressBar.hpp"
#include "LinSolve/SolverResult.hpp"
#include "LinSolve/linalg.hpp"
#include "LinSolve/numcepts/numcepts.hpp"
#include "LinSolve/TensorView/TensorView.hpp"

namespace linsol
{
  template <std::floating_point precision>
  struct cg_options
  {
    size_t maximum_iterations = 100;
    precision relative_tolerance = 1e-3;
    precision absolute_tolerance = 1e-6;
    precision maximum_seconds = std::numeric_limits<precision>::infinity();
    int verbose = 0;
  };

  /**
   * @brief Solves A * x == b with the conjugate gradient method.
   *
   * @tparam scalar float, double, complex<...>
   * @tparam Operator invocable
   * @tparam Preconditioner invocable
   * @param n number of unknowns (length of x and r)
   * @param x array of length n. On entry, x is an initial guess for the solution (or zeros). On exit, x is overwritten with the solution found by conjugate gradient.
   * @param A computes the action of A. Invocable like: A(const scalar *x, scalar *y). The result is y <- A * x.
   * @param r array of length n. On entry, r is the residual b - A * x for the initial x (or just b if x is initialized to zero). On exit, r is overwritten with the final residual.
   * @param M preconditioner. Invocable like: M(const scalar *x, scalar *y). The result is y <- M \ x. Where M approximates A.
   * @param options
   * @return SolverResult
   */
  template <numcepts::ScalarType scalar, typename Operator, typename Preconditioner>
  SolverResult cg(size_t n, scalar *_x, Operator &&A, scalar *_r, Preconditioner &&M, cg_options<numcepts::precision_t<scalar>> options = {})
  {
    using namespace tensor;
    using real = numcepts::precision_t<scalar>;

    SolverResult result;
    result.flag = 1;
    result.num_matvec = 0;
    result.residual_norm.reserve(options.maximum_iterations);
    result.time.reserve(options.maximum_iterations);

    Vector<scalar> p(n);
    Vector<scalar> z(n);
    Vector<scalar> Ap(n);
    auto r = reshape(_r, n);
    auto x = reshape(_x, n);

    M(r.data(), z.data());

    for (size_t i = 0; i < n; ++i)
      p[i] = z[i];

    real ρ = std::real(dot(r, z));
    if (ρ < 0)
    {
      result.flag = 4; // M not pos def
      if (options.verbose)
      {
        std::cout << "cg cannot proceed because the preconditioner is not positive definite." << std::endl;
      }
      return result;
    }

    real rnrm = std::sqrt(ρ);
    const real bnrm = rnrm;

    result.residual_norm.push_back((double)rnrm);
    result.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (rnrm < bnrm * options.relative_tolerance + options.absolute_tolerance)
    {
      result.flag = 0;

      if (options.verbose)
      {
        std::cout << "After 0 iterations, cg achieved a relative residual of " << rnrm / bnrm << std::endl;
        std::cout << "cg successfully converged within desired tolerance." << std::endl;
      }

      return result;
    }

    ProgressBar bar(options.maximum_iterations);
    if (options.verbose)
      std::cout << std::setprecision(5) << std::scientific;

    size_t it = 0;
    for (; it < options.maximum_iterations; ++it)
    {
      A(p.data(), Ap.data());
      result.num_matvec++;

      real δ = std::real(dot(p, Ap));

      if (δ < 0)
      {
        result.flag = 3; // A not pos def

        if (options.verbose)
          std::cout << "\ncg cannot proceed because A is not positive definite." << std::endl;

        return result;
      }

      real α = ρ / δ;

      for (size_t i = 0; i < n; ++i)
      {
        x[i] += α * p[i];
        r[i] -= α * Ap[i];
      }

      M(r.data(), z.data());

      real ρ1 = std::real(dot(r, z));

      if (ρ1 < 0)
      {
        result.flag = 4; // M not pos def

        if (options.verbose)
          std::cout << "\ncg cannot proceed because the preconditioner is not positive definite." << std::endl;
        return result;
      }

      real β = ρ1 / ρ;
      ρ = ρ1;

      for (size_t i = 0; i < n; ++i)
        p[i] = z[i] + β * p[i];

      rnrm = std::sqrt(ρ);
      result.residual_norm.push_back((double)rnrm);

      auto t1 = std::chrono::high_resolution_clock::now();
      double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      result.time.push_back(dur);

      if (options.verbose == 1)
      {
        ++bar;
        std::cout << bar.get() << " : (r, M \\ r) / ||b|| = " << std::setw(10) << rnrm / bnrm << "\r" << std::flush;
      }
      else if (options.verbose >= 2)
      {
        std::cout << "iteration " << std::setw(10) << it + 1 << " / " << options.maximum_iterations << " : (r, M \\ r) / ||b|| = " << std::setw(10) << rnrm / bnrm << std::endl;
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
      std::cout << "After " << it << " iterations, cg achieved a relative residual of " << rnrm / bnrm << std::endl;
      if (result)
        std::cout << "cg successfully converged within desired tolerance." << std::endl;
      else
        std::cout << "cg failed to converge within desired tolerance." << std::endl;
    }

    result.num_iter = it;
    return result;
  }

  /**
   * @brief Solves A * x == b with the conjugate gradient method.
   *
   * @tparam scalar float, double, complex<...>
   * @tparam Operator invocable
   * @param n number of unknowns (length of x and r).
   * @param x array of length n. On entry, x is an initial guess for the solution (or zeros). On exit, x is overwritten with the solution found by conjugate gradient.
   * @param A computes the action of A. Invocable like: A(const scalar *x, scalar *y). The result is y <- A * x.
   * @param r array of length n. On entry, r is the residual b - A * x for the initial x (or just b if x is initialized to zero). On exit, r is overwritten with the final residual.
   * @param options
   * @return SolverResult
   */
  template <numcepts::ScalarType scalar, typename Operator>
  SolverResult cg(size_t n, scalar *_x, Operator &&A, scalar *_r, cg_options<numcepts::precision_t<scalar>> options = {})
  {
    using namespace tensor;
    using real = numcepts::precision_t<scalar>;

    SolverResult result;
    result.flag = 1;
    result.num_matvec = 0;
    result.residual_norm.reserve(options.maximum_iterations);
    result.time.reserve(options.maximum_iterations);

    Vector<scalar> p(n);
    Vector<scalar> Ap(n);
    auto r = reshape(_r, n);
    auto x = reshape(_x, n);

    for (size_t i = 0; i < n; ++i)
      p[i] = r[i];

    real ρ = std::real(dot(r, r));
    real rnrm = std::sqrt(ρ);
    const real bnrm = rnrm;

    result.residual_norm.push_back((double)rnrm);
    result.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (rnrm < bnrm * options.relative_tolerance + options.absolute_tolerance)
    {
      result.flag = 0;

      if (options.verbose)
      {
        std::cout << "After 0 iterations, cg achieved a relative residual of " << rnrm / bnrm << std::endl;
        std::cout << "cg successfully converged within desired tolerance." << std::endl;
      }

      return result;
    }

    ProgressBar bar(options.maximum_iterations);
    if (options.verbose)
      std::cout << std::setprecision(5) << std::scientific;

    size_t it = 0;
    for (; it < options.maximum_iterations; ++it)
    {
      A(p.data(), Ap.data());
      result.num_matvec++;

      real δ = std::real(dot(p, Ap));

      if (δ < 0)
      {
        result.flag = 3; // A not pos def

        if (options.verbose)
          std::cout << "\ncg cannot proceed because A is not positive definite." << std::endl;

        return result;
      }

      real α = ρ / δ;

      for (size_t i = 0; i < n; ++i)
      {
        x[i] += α * p[i];
        r[i] -= α * Ap[i];
      }

      real ρ1 = std::real(dot(r, r));
      real β = ρ1 / ρ;
      ρ = ρ1;

      for (size_t i = 0; i < n; ++i)
        p[i] = r[i] + β * p[i];

      rnrm = std::sqrt(ρ);
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
      std::cout << "After " << it << " iterations, cg achieved a relative residual of " << rnrm / bnrm << std::endl;
      if (result)
        std::cout << "cg successfully converged within desired tolerance." << std::endl;
      else
        std::cout << "cg failed to converge within desired tolerance." << std::endl;
    }

    result.num_iter = it;
    return result;
  }

} // namespace linsol

#endif