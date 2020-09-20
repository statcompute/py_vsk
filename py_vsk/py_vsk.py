# py_vsk/py_vsk.py
# 0.0.1

from scipy.stats import norm
import scipy.optimize, numpy

def vsk_mle(x):
  """
  The function estimate Vasicek parameters by using MLE.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series
  Returns:
    A dictionary with parameters in the Vasicek distribution
  Example:
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  fn = lambda par: -sum(numpy.log(numpy.sqrt((1 - par[0]) / par[0]) * numpy.exp(
                   -1 / (2 * par[0]) * numpy.square(numpy.sqrt(1 - par[0]) * norm.ppf(_x) 
                   - norm.ppf(par[1])) + 1 / 2 * numpy.square(norm.ppf(_x)))))
  
  rs = scipy.optimize.minimize(fn, (0.1, numpy.mean(_x)), method = 'Nelder-Mead')
  
  return({"Rho": round(rs.x[0], 10), "P": round(rs.x[1], 10)})
  
  
def vsk_imm(x):
  """
  The function estimate Vasicek parameters by using indirect moment matching.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series
  Returns:
    A dictionary with parameters in the Vasicek distribution
  Example:
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  mu = numpy.mean(norm.ppf(_x))
  
  s2 = numpy.mean(numpy.square(norm.ppf(_x))) - mu * mu
   
  _p = norm.cdf(mu / numpy.sqrt(1 + s2))
  
  _r = s2 / (1 + s2)
  
  return({"Rho": round(_r, 10), "P": round(_p, 10)})  

