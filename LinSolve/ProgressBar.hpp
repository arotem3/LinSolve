#ifndef __LINSOLVE_PROGRESS_BAR_HPP__
#define __LINSOLVE_PROGRESS_BAR_HPP__

#include <string>
#include <sstream>
#include <iomanip>

namespace linsol
{

  class ProgressBar
  {
  private:
    int total;    // Total number of iterations
    int progress; // Current progress count
    int width;    // Width of the progress bar
    char fill;    // Character used to fill the bar

  public:
    ProgressBar(int iterations, int width = 30, char fill = '#')
        : total(iterations), progress(0), width(width), fill(fill) {}

    ProgressBar &operator++()
    {
      if (progress < total)
        ++progress;
      return *this;
    }

    // Get the current progress bar as a string
    std::string get() const
    {
      std::ostringstream oss;
      double r = (double)progress / total;
      int filledWidth = r * width;

      oss << "[";
      for (int i = 0; i < width; ++i)
        oss << ((i < filledWidth) ? fill : ' ');
      oss << "] ";

      int space = static_cast<int>(log10(total)) + 3;
      if (space < 10) // only print number of iterations if not too big
        oss << std::setw(space) << progress << " / " << total;

      int percentage = r * 100;
      oss << " ( " << percentage << "% )";

      return oss.str();
    }
  };
} // namespace linsol

#endif
