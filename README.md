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

Ce projet étudie la structure de dépendance entre plusieurs branches d'assurance non-vie à l'aide des copules et des vine copules.
L'objectif est d'estimer le besoin en fonds propres d'un assureur tout en tenant compte des dépendances non linéaires et des dépendances de queue entre les risques.

La méthodologie combine :
- L'ajustement des lois marginales
- L'analyse des dépendances via le Tau de Kendall et le Rho de Spearman
- La sélection de copules bivariées
- La modélisation par Vine Copule
- La simulation Monte Carlo
- L'estimation de la VaR et de la TVaR

Les données proviennent du package R `CASdatasets` et combinent :
- `freclaimset3multi9207`
- `freclaimset3fire9207`
- `freclaimset3dam9207`

Le jeu de données final contient des charges sinistres mensuelles agrégées sur la période 1992–2006.

---

## Repository Structure

```text
data/       -> datasets and processed data
figures/    -> generated plots and visualizations
report/     -> actuarial report and documentation
src/        -> Python source code
```

---

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

## 📊 Key Results / Résultats Clés

### 🇬🇧 Main Findings

- 180 monthly observations covering 8 insurance business lines
- 28 dependence pairs analyzed using Kendall's Tau and Spearman's Rho
- 14 statistically significant dependence structures detected
- Strongest dependence observed between:
  - Storm ↔ Property Damage (Kendall τ = 0.291)
  - Fire ↔ Large Fire Claims (Kendall τ = 0.246)

### Marginal Distribution Modeling

- Lognormal distribution selected for 7 out of 8 insurance branches
- Gamma distribution selected for Large Fire Claims
- Model selection performed using MLE, AIC, and Kolmogorov-Smirnov tests

### Copula Selection Results

| Pair Type | Selected Copula |
|---|---|
| Extreme upper-tail dependence | Gumbel |
| Lower-tail dependence | Clayton |
| Symmetric central dependence | Frank |
| Elliptical dependence | Gaussian |

The Gumbel copula was selected for Storm ↔ Property Damage and Property Damage ↔ Large Damage Claims, confirming upper-tail dependence during extreme climatic events.

### Vine Copula Model

- 8-dimensional Vine Copula — truncation level: 2
- Joint multivariate dependence modeled via pair-copula decomposition

### Monte Carlo Simulation

- 100,000 simulated scenarios
- Mean aggregated monthly loss: **49.89 M€**
- Monthly standard deviation: **21.93 M€**

### Capital Requirement Estimation

| Confidence Level | VaR (M€) | TVaR (M€) |
|---|---|---|
| 95% | 88.68 | 111.62 |
| 99% | 123.51 | 155.16 |
| 99.5% | 140.38 | 179.61 |
| 99.9% | 198.96 | 255.07 |

### Diversification Effect

Copula-based dependence modeling quantified diversification gains ranging from **12% at the 90% confidence level** up to **32% at the 99.9% confidence level**, demonstrating the importance of realistic dependence modeling in solvency capital estimation.

---

### 🇫🇷 Principaux Résultats

- 180 observations mensuelles couvrant 8 branches d'assurance
- 28 paires de dépendance analysées via le Tau de Kendall et le Rho de Spearman
- 14 dépendances statistiquement significatives détectées
- Les dépendances les plus fortes concernent :
  - Tempête ↔ Dommages (τ = 0.291)
  - Incendie ↔ Gros Incendies (τ = 0.246)

### Ajustement des Lois Marginales

- La loi Lognormale est retenue pour 7 garanties sur 8
- La loi Gamma est retenue pour les Gros Incendies
- Sélection réalisée via MLE, critère AIC et tests KS de Kolmogorov-Smirnov

### Résultats de Sélection des Copules

| Type de dépendance | Copule retenue |
|---|---|
| Dépendance de queue supérieure | Gumbel |
| Dépendance de queue inférieure | Clayton |
| Dépendance centrale symétrique | Frank |
| Dépendance elliptique | Gaussienne |

La copule de Gumbel est sélectionnée pour Tempête ↔ Dommages et Dommages ↔ Gros Dommages, confirmant la présence d'une dépendance de queue supérieure lors des événements climatiques extrêmes.

### Modèle Vine Copule

- Modèle Vine Copule de dimension 8 — niveau de troncature : 2
- Structure jointe multivariée modélisée via décomposition pair-copula

### Simulation Monte Carlo

- 100 000 scénarios simulés
- Charge mensuelle agrégée moyenne : **49.89 M€**
- Écart-type mensuel : **21.93 M€**

### Estimation du Besoin en Fonds Propres

| Niveau de confiance | VaR (M€) | TVaR (M€) |
|---|---|---|
| 95% | 88.68 | 111.62 |
| 99% | 123.51 | 155.16 |
| 99.5% | 140.38 | 179.61 |
| 99.9% | 198.96 | 255.07 |

### Effet de Diversification

La modélisation par copules met en évidence des gains de diversification allant de **12% au seuil 90%** jusqu'à **32% au seuil 99.9%**, soulignant l'importance d'une modélisation réaliste des dépendances dans l'estimation du capital de solvabilité.

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

## Data Source

Datasets originate from the `CASdatasets` package by Arthur Charpentier et al.
R preprocessing pipeline: `src/data_preprocessing.R`
Package: https://dutangc.perso.math.cnrs.fr/RRepository/pub/

---

## Academic Context

This project was developed as part of an actuarial and quantitative risk modeling study focused on multivariate insurance risk aggregation, dependence modeling, and solvency analysis under nonlinear dependence structures.
