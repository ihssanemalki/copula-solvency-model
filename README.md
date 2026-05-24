# Copula Solvency Model

Modeling multivariate insurance risks using copula theory and Monte Carlo simulation.

---

## 🇬🇧 English Description

This project studies the dependence structure between multiple non-life insurance branches using copula models and vine copulas.

The objective is to estimate the solvency capital requirement of an insurer while accounting for nonlinear and tail dependencies between risks.

The methodology combines:
- Marginal distribution fitting
- Dependence analysis using Kendall's Tau and Spearman's Rho
- Bivariate copula selection
- Vine copula modeling
- Monte Carlo simulation
- VaR and TVaR estimation

The dataset was generated in R using the `CASdatasets` package and combines:
- `freclaimset3multi9207`
- `freclaimset3fire9207`
- `freclaimset3dam9207`

The final dataset contains monthly aggregated insurance claims from 1992–2006.

---

## 🇫🇷 Description Française

Ce projet étudie la structure de dépendance entre plusieurs branches d’assurance non-vie à l’aide des copules et des vine copules.

L’objectif est d’estimer le besoin en fonds propres d’un assureur tout en tenant compte des dépendances non linéaires et des dépendances de queue entre les risques.

La méthodologie combine :
- L’ajustement des lois marginales
- L’analyse des dépendances via le Tau de Kendall et le Rho de Spearman
- La sélection de copules bivariées
- La modélisation par Vine Copule
- La simulation Monte Carlo
- L’estimation de la VaR et de la TVaR

Les données proviennent du package R `CASdatasets` et combinent :
- `freclaimset3multi9207`
- `freclaimset3fire9207`
- `freclaimset3dam9207`

Le jeu de données final contient des charges sinistres mensuelles agrégées sur la période 1992–2006. :contentReference[oaicite:0]{index=0}

---

## Repository Structure

```text
data/       -> datasets and processed data
figures/    -> generated plots and visualizations
report/     -> actuarial report and documentation
src/        -> Python source code
## Source Code

### `simulation_visualization.py`

Generates graphical visualizations related to:
- Dependence structures between insurance branches
- Copula behavior and tail dependence
- Monte Carlo simulated loss distributions
- Aggregated risk distributions
- VaR and TVaR graphical representations
- Diversification effects and solvency analysis plots

---

### `solvency_calculations.py`

Performs the numerical actuarial computations of the project, including:
- Marginal distribution fitting
- Dependence measurement using Kendall's Tau and Spearman's Rho
- Bivariate copula calibration and selection
- Vine copula construction
- Monte Carlo simulation
- Portfolio aggregation
- Value-at-Risk (VaR) estimation
- Tail-Value-at-Risk (TVaR) estimation
- Solvency capital requirement calculations

---

## Main Results

- 14 statistically significant dependence pairs identified
- Gumbel copulas selected for extreme-event insurance branches
- Vine copula model constructed for joint 8-risk dependence modeling
- 100,000 Monte Carlo simulations performed
- Solvency capital estimated using VaR and TVaR risk measures
- Diversification effects quantified through dependence modeling

---

## Tools & Libraries

### Python

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pyvinecopulib`

### R

- `CASdatasets`
- `copula`

---

## Academic Context

This project was developed as part of an actuarial and quantitative risk modeling study focused on multivariate insurance risk aggregation, dependence modeling, and solvency analysis under nonlinear dependence structures.
