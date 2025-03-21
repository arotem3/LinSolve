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
   * @brief Generalized minimal residual solver.
   *
   * @tparam scalar
   * @tparam BLASImpl implements BLAS operations. Has the same methods as linsol::BLAS.
   * This can be used to implement accelerated versions of the BLAS operations, e.g. using CUDA or OpenMP.
   * @tparam Allocator Allocator manages memory for the arrays used in the solver. This can be used to allocate memory on the GPU.
   */
  template <numcepts::ScalarType scalar, typename BLASImpl, typename Allocator>
  class gmres_solver
  {
  public:
    gmres_solver(gmres_options<numcepts::precision_t<scalar>> options = {})
        : options(options) {}

    void set_options(gmres_options<numcepts::precision_t<scalar>> options)
    {
      this->options = options;
    }

    void set_restart(size_t restart)
    {
      options.restart = restart;
    }

    void set_maximum_iterations(size_t maximum_iterations)
    {
      options.maximum_iterations = maximum_iterations;
    }

    void set_maximum_seconds(numcepts::precision_t<scalar> maximum_seconds)
    {
      options.maximum_seconds = maximum_seconds;
    }

    void set_relative_tolerance(numcepts::precision_t<scalar> relative_tolerance)
    {
      options.relative_tolerance = relative_tolerance;
    }

    void set_absolute_tolerance(numcepts::precision_t<scalar> absolute_tolerance)
    {
      options.absolute_tolerance = absolute_tolerance;
    }

    void set_verbose(int verbose)
    {
      options.verbose = verbose;
    }

    /**
     * @brief solves A * x = b using the generalized minimal residual method.
     *
     * @tparam Operator invocable
     * @param n number of unknowns. Length of x and b.
     * @param x array of length n. On entry, initial guess. On exit, solution.
     * @param A invocable object that computes A * x. Signature: void A(const scalar *x, scalar *y) such that y = A * x.
     * @param b array of length n. The right hand side b.
     * @return SolverResult
     */
    template <typename Operator>
    SolverResult solve(size_t n, scalar *x, Operator &&A, const scalar *b)
    {
      using namespace tensor;
      using real = numcepts::precision_t<scalar>;
      constexpr scalar one(1), zero(0);

      const size_t m = options.restart;
      const size_t m1 = m + 1;

      real bnrm = blas.norm(n, b, 1);

      V.reshape(n, m1);
      H.reshape(m1, m);
      sn.reshape(m);
      cs.reshape(m);
      η.reshape(m1);

      auto r = reshape(V.data(), n); // r is the first column of V.

      SolverResult result;
      result.residual_norm.reserve(options.maximum_iterations * options.restart + 1);
      result.time.reserve(options.maximum_iterations + 1);
      result.num_iter = 0;
      result.num_matvec = 0;
      result.flag = 1;

      A(x, r.data());
      result.num_matvec++;

      blas.scal(n, -one, r.data(), 1);
      blas.axpy(n, one, b, 1, r.data(), 1);

      real rnrm = blas.norm(n, r.data(), 1);

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
        blas.scal(n, β, r.data(), 1);

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
            scalar *vj = V.data() + j * n;
            scalar d = blas.dot(n, vj, 1, vk1, 1);
            H(j, k) = d;

            blas.axpy(n, -d, vj, 1, vk1, 1);
          }

          H(k1, k) = blas.norm(n, vk1, 1);

          if (H(k1, k) == zero)
            break;

          β = one / H(k1, k);
          blas.scal(n, β, vk1, 1);

          givens_rotation(H(all(), k), cs, sn, k);
          η(k1) = -sn(k) * η(k);
          η(k) = cs(k) * η(k);

          rnrm = std::abs(η(k1));
          result.residual_norm.push_back((double)rnrm);

          if (rnrm < options.relative_tolerance * bnrm + options.absolute_tolerance)
            break;
        }

        triu_solve(k1, H, η);
        for (size_t k = 0; k < k1; ++k)
        {
          scalar *vk = V.data() + k * n;
          blas.axpy(n, η(k), vk, 1, x, 1);
        }

        A(x, r.data());
        result.num_matvec++;

        blas.scal(n, -one, r.data(), 1);
        blas.axpy(n, one, b, 1, r.data(), 1);

        rnrm = blas.norm(n, r.data(), 1);
        result.residual_norm.back() = (double)rnrm; // more numerically stable than computing from η.

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
     * @brief Solves A * x = b using the generalized minimal residual method with preconditioner M.
     *
     * @tparam Operator invocable
     * @tparam Precond invocable
     * @param n number of unknowns. Length of x and r.
     * @param x array of length n. On entry, initial guess. On exit, solution.
     * @param A invocation object that computes A * x. Signature: void A(const scalar *x, scalar *y) such that y = A * x.
     * @param b array of length n. The right hand side b.
     * @param M preconditioner. Invocable object that solves M y == x. Signature: void M(const scalar *x, scalar *y) such that y = M \ x.
     * @return SolverResult
     */
    template <typename Operator, typename Precond>
    SolverResult solve(size_t n, scalar *x, Operator &&A, const scalar *b, Precond &&M)
    {
      tensor::Vector<scalar, Allocator> y(n);
      tensor::Vector<scalar, Allocator> r0(n);

      auto MA = [&](const scalar *u, scalar *v) mutable
      {
        A(u, y.data());
        M(y.data(), v);
      };

      M(b, r0.data());

      return solve(n, x, MA, r0.data());
    }

  private:
    using real = numcepts::precision_t<scalar>;

    tensor::Matrix<scalar, Allocator> V;
    tensor::Matrix<scalar> H;
    tensor::Vector<scalar> sn;
    tensor::Vector<scalar> cs;
    tensor::Vector<scalar> η;

    gmres_options<real> options;

    BLASImpl blas;
  };

  /**
   * @brief solves A * x = b using the generalized minimal residual method.
   *
   * @tparam scalar float, double, complex<float>, complex<double>
   * @tparam Operator invocable
   * @param n number of unkowns. Length of x and b.
   * @param x array of length n. On entry, initial guess. On exit, solution.
   * @param A invocable object that computes A * x. Signature: void A(const scalar *x, scalar *y) such that y = A * x.
   * @param b array of length n. The right hand side b.
   * @param options
   * @return SolverResult
   */
  template <numcepts::ScalarType scalar, typename Operator>
  SolverResult gmres(size_t n, scalar *_x, Operator &&A, const scalar *_b, gmres_options<numcepts::precision_t<scalar>> options = {})
  {
    gmres_solver<scalar, BLAS, std::allocator<scalar>> solver(options);
    return solver.solve(n, _x, A, _b);
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
    gmres_solver<scalar, BLAS, std::allocator<scalar>> solver(options);
    return solver.solve(n, x, A, b, M);
  }
} // namespace linsol

#endif