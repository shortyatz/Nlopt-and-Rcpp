// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include "RcppArmadillo.h"
#include <nlopt.hpp>
#include <vector>
#include <random>
#include <math.h>

using namespace arma;

struct my_constraint_data
{
  double a;
  double b;
};

double myvfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
  if (grad) {
    grad[0] = 0.0;
    grad[1] = 0.5 / sqrt(x[1]);
  }
  return sqrt(x[1]);
}


double myvconstraint(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_constraint_data *d = reinterpret_cast<my_constraint_data*>(data);
  double a = d->a, b = d->b;
  if (!grad.empty()) {
    grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
    grad[1] = -1.0;
  }
  return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

// [[Rcpp::export]]
double test()
{
  nlopt::opt opt(nlopt::LD_MMA, 2);
  std::vector<double> lb(2);
  lb[0] = R_NegInf;
  lb[1] = 0.0;
  opt.set_lower_bounds(lb);
  opt.set_min_objective(myvfunc, NULL);
  my_constraint_data data[2] = { {2,0}, {-1,1} };
//  opt.add_inequality_constraint(myvconstraint, &data[0], 1e-8);
//  opt.add_inequality_constraint(myvconstraint, &data[1], 1e-8);

  opt.set_xtol_rel(1e-4);
  std::vector<double> x(2);
  x[0] = 1.234;
  x[1] = 5.678;
  double minf;
  nlopt::result result = opt.optimize(x, minf);
  Rcpp::Rcout << int(result) << "\n";
  Rcpp::Rcout << x[0] << "," << x[1] << "\n";
  return minf;
}

