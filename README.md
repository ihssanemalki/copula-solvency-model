# Copula Solvency Model

Modeling multivariate insurance risks using copula theory and Monte Carlo simulation.

## Project Overview

This project studies the dependence structure between multiple non-life insurance branches using copula models and vine copulas.

The objective is to estimate the solvency capital requirement of an insurer while accounting for nonlinear and tail dependencies between risks.

The methodology combines:
- Marginal distribution fitting
- Dependence analysis using Kendall's Tau and Spearman's Rho
- Bivariate copula selection
- Vine copula modeling
- Monte Carlo simulation
- VaR and TVaR estimation

## Dataset

The dataset was generated in R using the CASdatasets package and combines:
- freclaimset3multi9207
- freclaimset3fire9207
- freclaimset3dam9207

The final dataset contains monthly aggregated insurance claims from 1992–2006.

## Repository Structure

```text
data/       -> datasets and processed data
figures/    -> generated plots and visualizations
report/     -> actuarial report and documentation
src/        -> Python source code
