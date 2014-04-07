//  Example for using rtnorm
//  
//  Copyright (C) 2012 Guillaume Dollé, Vincent Mazet (LSIIT, CNRS/Université de Strasbourg)
//  Licence: GNU General Public License Version 2
//  see http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
//
//  Depends: LibGSL
//  OS: Unix based system


#include <unistd.h>
#include <iostream>
#include <gsl/gsl_rng.h>
#include "rtnorm.hpp"
#include "call_rtnorm.hpp"

void call_rtnorm(double result[], int K, double a, double b, double mu, double sigma)
{
  std::pair<double, double> s;  // Output argument of rtnorm
  long seed;

  //--- GSL random init ---
  gsl_rng_env_setup();                          // Read variable environnement
  const gsl_rng_type* type = gsl_rng_default;   // Default algorithm 'twister'
  gsl_rng *gen = gsl_rng_alloc (type);          // Rand generator allocation

  seed = time(NULL) + clock() + random();
  gsl_rng_set (gen, seed);

  //--- generate the random numbers ---
  for(int k=0; k<K; k++)
  {
    s = rtnorm(gen,a,b,mu,sigma);
    result[k] = s.first;
  }

  gsl_rng_free(gen);                            // GSL rand generator deallocation

  return;
}