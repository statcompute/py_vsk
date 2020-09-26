
### <p align="center"> A Collection of Python Functions </p>
### <p align="center"> for Vasicek Distribution (PY_VSK) </p>

#### Introduction

In the context of statistics, the Vasicek distribution is a continuous distribution governing the open interval between 0 and 1 with two parameters, namely Rho and P, which is similar to the Beta and Kumaraswamy distributions.

The Vasicek distribution has often been used to describe the portfolio credit loss in the development of Economic Capital models.

#### Core Functions

```
py_vsk
   |-- vsk_mle() : Estimate Vasicek parameters by using MLE.
   |-- vsk_imm() : Estimate Vasicek parameters by using indirect moment matching.
   |-- vsk_pdf() : Calculates the probability density function of Vasicek.
   |-- vsk_cdf() : Calculates the probability cumulative function of Vasicek.
   |-- vsk_ppf() : Calculates the percentile point function (CDF inverse) of Vasicek.
   `-- vsk_rvs() : Generates random numbers following the Vasicek distribution
```
