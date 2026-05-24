# ============================================================
# DATA PREPROCESSING SCRIPT
# ============================================================
# This script installs and loads the CASdatasets package,
# extracts the insurance datasets used in the project,
# and exports them into CSV format for Python-based
# copula modeling and Monte Carlo simulation.
# ============================================================

# Install CASdatasets package from source
install.packages(
  "C:/Users/HP/Downloads/CASdatasets_1.2-1.tar.gz",
  repos = NULL,
  type = "source"
)

# Load package
library(CASdatasets)

# Load datasets
data(freclaimset3multi9207)
data(freclaimset3fire9207)
data(freclaimset3dam9207)

# Export datasets to CSV
write.csv(
  freclaimset3multi9207,
  "C:/Users/HP/Downloads/freclaimset3multi9207.csv",
  row.names = FALSE
)

write.csv(
  freclaimset3fire9207,
  "C:/Users/HP/Downloads/freclaimset3fire9207.csv",
  row.names = FALSE
)

write.csv(
  freclaimset3dam9207,
  "C:/Users/HP/Downloads/freclaimset3dam9207.csv",
  row.names = FALSE
)

# ============================================================
# Source:
# https://dutangc.perso.math.cnrs.fr/RRepository/pub/src/contrib/CASdatasets_1.2-1.tar.gz
# ============================================================
