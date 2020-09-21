# py_vsk/py_vsk.py
# 0.0.2

from scipy.stats import norm
import scipy.optimize, numpy

########## 01. vsk_mle() ########## 

def vsk_mle(x):
  """
  The function estimate Vasicek parameters by using MLE.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series
  Returns:
    A dictionary with parameters in the Vasicek distribution
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  fn = lambda par: -sum(numpy.log(numpy.sqrt((1 - par[0]) / par[0]) * numpy.exp(
                   -1 / (2 * par[0]) * numpy.square(numpy.sqrt(1 - par[0]) * norm.ppf(_x) 
                   - norm.ppf(par[1])) + 1 / 2 * numpy.square(norm.ppf(_x)))))
  
  rs = scipy.optimize.minimize(fn, (0.1, numpy.mean(_x)), method = 'Nelder-Mead')
  
  return({"Rho": round(rs.x[0], 10), "P": round(rs.x[1], 10)})
  
  
########## 02. vsk_imm() ########## 

def vsk_imm(x):
  """
  The function estimate Vasicek parameters by using indirect moment matching.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series
  Returns:
    A dictionary with parameters in the Vasicek distribution
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  mu = numpy.mean(norm.ppf(_x))
  
  s2 = numpy.mean(numpy.square(norm.ppf(_x))) - mu * mu
   
  _p = norm.cdf(mu / numpy.sqrt(1 + s2))
  
  _r = s2 / (1 + s2)
  
  return({"Rho": round(_r, 10), "P": round(_p, 10)})  


########## 03. vsk_pdf() ########## 

def vsk_pdf(x, Rho, P):
  """
  The function calculates the probability density function of Vasicek.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series;
    Rho : The Rho parameter in the Vasicek distribution
    P   : The P parameter in the Vasicek distribution
  Returns:
    A list of dictionaries with each value in the x and the corresponding pdf.
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  fn = lambda par: numpy.sqrt((1 - par[0]) / par[0]) * numpy.exp(
                   -1 / (2 * par[0]) * numpy.square(numpy.sqrt(1 - par[0]) * norm.ppf(_x) 
                   - norm.ppf(par[1])) + 1 / 2 * numpy.square(norm.ppf(_x)))
  
  return([{"x": _[0], "pdf": _[1]} for _ in zip(_x, fn([Rho, P]))])


########## 04. vsk_ppf() ##########

def vsk_ppf(Alpha, Rho, P):
  """
  The function calculates the percentile point function (CDF inverse) of Vasicek.
  Parameters:
    Alpha : A numeric vector of probabilities 
    Rho   : The Rho parameter in the Vasicek distribution
    P     : The P parameter in the Vasicek distribution
  Returns:
    A list of dictionaries with each value in the Alpha and the corresponding ppf.
  """  
  
  _a = [_ for _ in Alpha if _ > 0 and _ < 1 and not numpy.isnan(_)]

  _p = norm.cdf((norm.ppf(P) + numpy.sqrt(Rho) * norm.ppf(_a)) / numpy.sqrt(1 - Rho))
                
  return([{"Alpha": _[0], "ppf": round(_[1], 10)} for _ in zip(_a, _p)])


########## 05. vsk_rvs() ##########

def vsk_rvs(n, Rho, P, seed = 1):
  """
  The function generates random numbers following the Vasicek distribution with
  parameter Rho and P.
  Parameters:
    n    : The number of observations
    Rho  : The Rho parameter in the Vasicek distribution
    P    : The P parameter in the Vasicek distribution
    seed : The seed value used to generate random numbers.
  Returns:
    A list of random numbers under the Vasicek distributional assumption.
  """

  rn = norm.rvs(size = n, random_state = seed)

  rv = norm.cdf((norm.ppf(P) - numpy.sqrt(Rho) * rn) / numpy.sqrt(1 - Rho))

  return([_ for _ in rv])

