
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
   |-- vsk_cdf()   : Calculates the cumulative distribution function of Vasicek.
   |-- vsk_ppf()   : Calculates the percentile point function (CDF inverse) of Vasicek.
   |-- vsk_rvs()   : Generates random numbers following the Vasicek distribution
   |-- gof_ks()    : Performs the Kolmogorov-Smirnov GoF test for the Vasicek distribution
   `-- gof_chisq() : Performs the Chi-Square GoF test for the Vasicek distribution
```

#### Example

While default rates cannot be predicted with certainty, the probability of future default rate can be assessed based on the appropriate statistical distribution. 

Below is the list of default rates for 100 largest banks in the last 15 years (https://www.federalreserve.gov/releases/chargeoff/deltop100nsa.htm). 

```python
df = [0.0171, 0.0214, 0.0275, 0.0317, 0.0400, 0.0533, 0.0692, 0.0901, 0.0984, 0.1051, 
      0.1117, 0.0684, 0.0317, 0.0190, 0.0158]
```
Based on the above, we can estimate parameters of the corresponding Vasicek distribution.
```python
import py_vsk
py_vsk.vsk_mle(df)
# {'Rho': 0.0937789425, 'P': 0.0532449939}
```
While the default rate reached the highest of 11.17% in 2009, this default rate is equivalent to ~93%ile in the Vasicek distribution. 
```python
py_vsk.vsk_cdf([max(df)], Rho = 0.0938, P = 0.0532)
# [{'x': 0.1117, 'cdf': 0.9315524265618389}]
```
In addition, the result below shows that there is an 1-in-100 chance that the default rate could be as high as 17.16%. 
```python
py_vsk.vsk_ppf([0.99], Rho = 0.0938, P = 0.0532)
# [{'Alpha': 0.99, 'ppf': 0.17165620222653788}]
```

#### Reference

Tasche, Dirk. (2008). The Vasicek Distribution.
