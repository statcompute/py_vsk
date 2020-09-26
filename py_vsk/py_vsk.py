# py_vsk/py_vsk.py
# 0.0.3

from scipy.stats import norm
import scipy.optimize, numpy

########## 01. vsk_mle() ########## 

def vsk_mle(x):
  """
  The function estimates Vasicek parameters by using MLE.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series
  Returns:
    A dictionary with parameters in the Vasicek distribution
  Example:
    import py_vsk
    x = py_vsk.vsk_rvs(1000, Rho = 0.2, P = 0.3)
    py_vsk.vsk_mle(x)
    # {'Rho': 0.1938984897, 'P': 0.2932844544}
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
  The function estimates Vasicek parameters by using indirect moment matching.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series
  Returns:
    A dictionary with parameters in the Vasicek distribution
  Example:
    import py_vsk
    x = py_vsk.vsk_rvs(1000, Rho = 0.2, P = 0.3)
    py_vsk.vsk_imm(x)
    # {'Rho': 0.1939333994, 'P': 0.2932867178}
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
  Example:
    import py_vsk
    py_vsk.vsk_pdf([0.01, 0.02], Rho = 0.2, P = 0.3)
    # [{'x': 0.01, 'pdf': 0.07019659048697276},
    #  {'x': 0.02, 'pdf': 0.22207563838880806}]
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  fn = lambda par: numpy.sqrt((1 - par[0]) / par[0]) * numpy.exp(
                   -1 / (2 * par[0]) * numpy.square(numpy.sqrt(1 - par[0]) * norm.ppf(_x) 
                   - norm.ppf(par[1])) + 1 / 2 * numpy.square(norm.ppf(_x)))
  
  return([{"x": _[0], "pdf": _[1]} for _ in zip(_x, fn([Rho, P]))])


########## 04. vsk_cdf() ########## 

def vsk_cdf(x, Rho, P):
  """
  The function calculates the cumulative density function of Vasicek.
  Parameters:
    x   : A numeric vector in the interval of (0, 1), which can be a list, 
          1-D numpy array, or pandas series;
    Rho : The Rho parameter in the Vasicek distribution
    P   : The P parameter in the Vasicek distribution
  Returns:
    A list of dictionaries with each value in the x and the corresponding cdf.
  Example:
    import py_vsk
    vsk_cdf([0.278837772815679, 0.5217229060260343], Rho = 0.2, P = 0.3)
    # [{'x': 0.278837772815679, 'cdf': 0.5},
    #  {'x': 0.5217229060260343, 'cdf': 0.8999999999999999}] 
  """  
  
  _x = [_ for _ in x if _ > 0 and _ < 1 and not numpy.isnan(_)]

  fn = lambda par: norm.cdf((numpy.sqrt(1 - par[0]) * norm.ppf(_x) - norm.ppf(par[1])) 
                   / numpy.sqrt(par[0]))
  
  return([{"x": _[0], "cdf": _[1]} for _ in zip(_x, fn([Rho, P]))])


########## 05. vsk_ppf() ##########

def vsk_ppf(Alpha, Rho, P):
  """
  The function calculates the percentile point function (CDF inverse) of Vasicek.
  Parameters:
    Alpha : A numeric vector of probabilities 
    Rho   : The Rho parameter in the Vasicek distribution
    P     : The P parameter in the Vasicek distribution
  Returns:
    A list of dictionaries with each value in the Alpha and the corresponding ppf.
  Example:
    import py_vsk
    py_vsk.vsk_ppf([0.5, 0.9], Rho = 0.2, P = 0.3)
    # [{'Alpha': 0.5, 'ppf': 0.278837772815679}, 
    #  {'Alpha': 0.9, 'ppf': 0.5217229060260343}]
  """  
  
  _a = [_ for _ in Alpha if _ > 0 and _ < 1 and not numpy.isnan(_)]

  _p = norm.cdf((norm.ppf(P) + numpy.sqrt(Rho) * norm.ppf(_a)) / numpy.sqrt(1 - Rho))
                
  return([{"Alpha": _[0], "ppf": _[1]} for _ in zip(_a, _p)])


########## 06. vsk_rvs() ##########

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
  Example:
    import py_vsk
    x = py_vsk.vsk_rvs(1000, Rho = 0.2, P = 0.3)
  """

  rn = norm.rvs(size = n, random_state = seed)

  rv = norm.cdf((norm.ppf(P) - numpy.sqrt(Rho) * rn) / numpy.sqrt(1 - Rho))

  return([_ for _ in rv])

