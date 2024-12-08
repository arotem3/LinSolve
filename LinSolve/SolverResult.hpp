#ifndef __LINSOLVE_SOLVER_RESULT_HPP__
#define __LINSOLVE_SOLVER_RESULT_HPP__

#include <vector>

namespace linsol
{
  struct SolverResult
  {
    bool success;                      // True if solver converged within tolerance.
    int num_iter;                      // Total number of iterations required to converge.
    int num_matvec;                    // Total number of matrix-vector multiplications.
    std::vector<double> residual_norm; // The norm of the residual at each iteration.
    std::vector<double> time;          // The total computation time to each iteration.
  };

} // namespace linsol

#endif
