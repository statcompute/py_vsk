
### <p align="center"> A Collection of Python Functions </p>
### <p align="center"> for Vasicek Distribution (PY_VSK) </p>

#### Introduction

In the context of statistics, the Vasicek distribution is a continuous distribution governing the open interval between 0 and 1 with two parameters, namely Rho and P, which is similar to the Beta and Kumaraswamy distributions. 

The Vasicek distribution has often been used to describe the portfolio credit loss in the development of Economic Capital models. The py\_vsk package is a collection of miscellaneous python functions related to the Vasicek distribution with the intent to make the lives of risk modelers easier.


#### Core Functions

```
py_vsk
  |-- vsk_mle()   : Estimates Vasicek parameters by using MLE.
  |-- vsk_imm()   : Estimates Vasicek parameters by using indirect moment matching.
  |-- vsk_dmm()   : Estimates Vasicek parameters by using direct moment matching.
  |-- vsk_qbe()   : Estimates Vasicek parameters by using quantile-based estimator.
  |-- vsk_pdf()   : Calculates the probability density function of Vasicek.
  |-- vsk_cdf()   : Calculates the probability cumulative function of Vasicek.
  |-- vsk_ppf()   : Calculates the percentile point function (CDF inverse) of Vasicek.
  |-- vsk_rvs()   : Generates random numbers following the Vasicek distribution.
  |-- get_Rho()   : Solves for the Rho parameter such that vsk_cdf(x, Rho, P) = Alpha.
  |-- gof_ks()    : Performs the Kolmogorov-Smirnov GoF test for the Vasicek distribution.
  `-- gof_chisq() : Performs the Chi-Square GoF test for the Vasicek distribution.
```

#### Example

While future default rates cannot be predicted with certainty, the probability of default rates can be assessed based on the appropriate statistical distribution. 

Below is the list of delinquent rates for 100 largest banks in the last 20 years (https://www.federalreserve.gov/releases/chargeoff/deltop100nsa.htm). 
```python
df = [0.0171, 0.0214, 0.0275, 0.0317, 0.0400, 0.0533, 0.0692, 0.0901, 0.0984, 0.1051, 
      0.1117, 0.0684, 0.0317, 0.0190, 0.0158, 0.0139, 0.0179, 0.0200, 0.0241, 0.0264]
```
Based on the above, we can estimate parameters of the corresponding Vasicek distribution. While the P parameter shows the long-term average of delinquent rates, the Rho parameter describes the degree of association with systematic risk factors.  
```python
import py_vsk
py_vsk.vsk_mle(df)
# {'Rho': 0.0939762321, 'P': 0.0446574471}
```
The delinquent rate reached the highest of 11.17% in 2009 and is equivalent to ~96%ile in the corresponding Vasicek distribution, suggesting that the 2009 downturn is an 1-in-25 event.
```python
py_vsk.vsk_cdf([max(df)], 0.0939762321, 0.0446574471)
# [{'x': 0.1117, 'cdf': 0.9609532701414676}]
```
In addition, the result below shows that there is an 1% chance that the delinquent rate could be as high as 15%. 
```python
py_vsk.vsk_ppf([0.99], 0.0939762321, 0.0446574471)
# [{'Alpha': 0.99, 'ppf': 0.15016266823403973}]
```

#### Reference

Tasche, Dirk. (2008). The Vasicek Distribution.
