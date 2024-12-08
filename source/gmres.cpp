#include "LinSolve/gmres.hpp"

using zdouble = std::complex<double>;
using zfloat = std::complex<float>;
using namespace tensor;

extern "C" void dtrsv_(char *uplo, char *trans, char *diag, int *n, const double *a, int *lda, double *x, int *inc_x);
extern "C" void strsv_(char *uplo, char *trans, char *diag, int *n, const float *a, int *lda, float *x, int *inc_x);
extern "C" void ctrsv_(char *uplo, char *trans, char *diag, int *n, const zfloat *a, int *lda, zfloat *x, int *inc_x);
extern "C" void ztrsv_(char *uplo, char *trans, char *diag, int *n, const zdouble *a, int *lda, zdouble *x, int *inc_x);

namespace linsol
{
  void triu_solve(size_t n, const Matrix<double> &H, Vector<double> &η)
  {
    int N = n;
    int one = 1;
    int lda = H.shape(0);
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";

    dtrsv_(uplo, trans, diag, &N, H.data(), &lda, η.data(), &one);
  }

  void triu_solve(size_t n, const Matrix<float> &H, Vector<float> &η)
  {
    int N = n;
    int one = 1;
    int lda = H.shape(0);
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";

    strsv_(uplo, trans, diag, &N, H.data(), &lda, η.data(), &one);
  }

  void triu_solve(size_t n, const Matrix<zfloat> &H, Vector<zfloat> &η)
  {
    int N = n;
    int one = 1;
    int lda = H.shape(0);
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";

    ctrsv_(uplo, trans, diag, &N, H.data(), &lda, η.data(), &one);
  }

  void triu_solve(size_t n, const Matrix<zdouble> &H, Vector<zdouble> &η)
  {
    int N = n;
    int one = 1;
    int lda = H.shape(0);
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";

    ztrsv_(uplo, trans, diag, &N, H.data(), &lda, η.data(), &one);
  }
} // namespace linsol
