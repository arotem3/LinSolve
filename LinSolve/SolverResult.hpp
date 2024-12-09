#ifndef __LINSOLVE_SOLVER_RESULT_HPP__
#define __LINSOLVE_SOLVER_RESULT_HPP__

#include <vector>

namespace linsol
{
  struct SolverResult
  {
    int flag;                          // Success flag: 0 - solver converged. Anything else depends on the solver.
    int num_iter;                      // Total number of iterations required to converge.
    int num_matvec;                    // Total number of matrix-vector multiplications.
    std::vector<double> residual_norm; // The norm of the residual at each iteration.
    std::vector<double> time;          // The total computation time to each iteration.

    operator bool() const
    {
      return (flag == 0);
    }
  };

} // namespace linsol

#endif
