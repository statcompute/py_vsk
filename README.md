
### <p align="center"> A Collection of Python Functions </p>
### <p align="center"> for Vasicek Distribution (PY_VSK) </p>

#### Introduction

In the context of statistics, the Vasicek distribution is a continuous distribution governing the open interval between 0 and 1 with two parameters, namely Rho and P, which is similar to the Beta and Kumaraswamy distributions. 

The Vasicek distribution has often been used to describe the portfolio credit loss in the development of Economic Capital models. The py\_vsk package is a collection of miscellaneous python functions related to the Vasicek distribution with the intent to make the lives of risk modelers easier.


#### Core Functions

```
py_vsk
   |-- vsk_mle()   : Estimate Vasicek parameters by using MLE.
   |-- vsk_imm()   : Estimate Vasicek parameters by using indirect moment matching.
   |-- vsk_dmm()   : Estimate Vasicek parameters by using direct moment matching.
   |-- vsk_qbe()   : Estimate Vasicek parameters by using quantile-based estimator.
   |-- vsk_pdf()   : Calculates the probability density function of Vasicek.
   |-- vsk_cdf()   : Calculates the cumulative distribution function of Vasicek.
   |-- vsk_ppf()   : Calculates the percentile point function (CDF inverse) of Vasicek.
   |-- vsk_rvs()   : Generates random numbers following the Vasicek distribution
   |-- gof_ks()    : Performs the Kolmogorov-Smirnov goodness-of-fit test for the Vasicek distribution
   `-- gof_chisq() : Performs the Chi-Square goodness-of-fit test for the Vasicek distribution
```
