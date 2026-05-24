"""
Pipeline Cadoux v2 — 8 garanties, données françaises réelles
=============================================================
Datasets combinés (CASdatasets) :
  - freclaimset3multi9207 : 6 garanties agrégées (1992-2006)
  - freclaimset3fire9207  : sinistres individuels incendie (gros sinistres)
  - freclaimset3dam9207   : sinistres individuels dommages (gros sinistres)

Résultat : 180 mois × 8 garanties → 28 paires à analyser
Méthode  : Cadoux & Loizeau (2004)
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools, warnings
warnings.filterwarnings('ignore')
import pyvinecopulib as pv

np.random.seed(42)

# ── chemin automatique (même dossier que le script) ──────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "C:\\Users\\HP\\Downloads\\monthly_claims_combined.csv")
OUT_PNG    = os.path.join(SCRIPT_DIR, r"C:\Users\HP\Downloads\cadoux_v2_results.png")

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────
print("=" * 64)
print("ÉTAPE 1 — DONNÉES (8 garanties, assureur français non-vie)")
print("=" * 64)

df = pd.read_csv(CSV_PATH)
df['Mois'] = pd.to_datetime(df['Mois'])
df = df.sort_values('Mois').reset_index(drop=True)

GARANTIES = {
    'Fire_Claim'      : 'Incendie (total)',
    'HSS_Claim'       : 'Tempête',
    'Damage_Claim'    : 'Dommages (total)',
    'Thief_Claim'     : 'Vol',
    'TPL_Claim'       : 'RC (TPL)',
    'Other_Claim'     : 'Autres',
    'LargeFire_Claim' : 'Gros Incendies',
    'LargeDam_Claim'  : 'Gros Dommages',
}
cols   = list(GARANTIES.keys())
labels = list(GARANTIES.values())

print(f"  Période   : {df['Mois'].min().strftime('%Y-%m')} → {df['Mois'].max().strftime('%Y-%m')}")
print(f"  Mois      : {len(df)}")
print(f"  Garanties : {len(cols)}  →  {len(cols)*(len(cols)-1)//2} paires à analyser")
print(f"\n  Charge mensuelle moyenne (€) :")
for c, l in GARANTIES.items():
    print(f"    {l:22s}: {df[c].mean():>15,.0f} €")

# ─────────────────────────────────────────────────────────────
# 2. REDRESSEMENT INFLATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 2 — REDRESSEMENT INFLATION (2% / an, base 2006)")
print("=" * 64)

base_year = 2006
inflation = 0.02
df['annee'] = df['Mois'].dt.year
for c in cols:
    df[c + '_adj'] = df[c] * (1 + inflation) ** (base_year - df['annee'])

cols_adj = [c + '_adj' for c in cols]
print(f"  Facteur 1992 : ×{(1.02)**(base_year-1992):.3f}  |  Facteur 2005 : ×{(1.02)**(base_year-2005):.3f}")

# ─────────────────────────────────────────────────────────────
# 3. AJUSTEMENT LOIS MARGINALES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 3 — AJUSTEMENT LOIS MARGINALES")
print("=" * 64)

DISTS = {'Lognormale': stats.lognorm,
         'Weibull'   : stats.weibull_min,
         'Gamma'     : stats.gamma}

best_fits = {}
summary_fits = []

for c, label in zip(cols_adj, labels):
    x   = df[c].values
    res = {}
    for dname, dist in DISTS.items():
        params  = dist.fit(x, floc=0)
        ll      = np.sum(dist.logpdf(x, *params))
        aic     = 2*len(params) - 2*ll
        ks, kp  = stats.kstest(x, lambda v: dist.cdf(v, *params))
        res[dname] = {'params': params, 'aic': aic, 'ks_p': kp}
    best = min(res, key=lambda d: res[d]['aic'])
    best_fits[c] = {'dist': best, 'params': res[best]['params']}
    summary_fits.append({'Garantie': label, 'Loi retenue': best,
                         'KS p-value': round(res[best]['ks_p'], 3)})
    print(f"  {label:22s} → {best:12s}  (KS p={res[best]['ks_p']:.3f})", end="")
    print(f"  [AIC: " + " | ".join(f"{d}={r['aic']:.0f}" for d,r in res.items()) + "]")

# ─────────────────────────────────────────────────────────────
# 4. ANALYSE DES DÉPENDANCES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 4 — ANALYSE DES DÉPENDANCES  (28 paires)")
print("=" * 64)

X     = df[cols_adj].values
pairs = list(itertools.combinations(range(len(cols_adj)), 2))
pair_labels = [f"{labels[i]} ↔ {labels[j]}" for i,j in pairs]

dep_results = []
for (i,j), plabel in zip(pairs, pair_labels):
    tau, p_tau = kendalltau(X[:,i], X[:,j])
    rho, p_rho = spearmanr(X[:,i], X[:,j])
    dep_results.append({'Paire': plabel,
                        'Kendall τ': round(tau,4),
                        'p-val'    : round(p_tau,4),
                        'Spearman ρ': round(rho,4),
                        'Sig.'     : '✓' if p_tau < 0.05 else '✗'})

dep_df  = pd.DataFrame(dep_results)
sig_df  = dep_df[dep_df['Sig.']=='✓'].sort_values('Kendall τ', ascending=False)

print(f"\n  Paires significatives : {len(sig_df)} / {len(dep_df)}\n")
print(sig_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 5. PSEUDO-UNIFORMES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 5 — TRANSFORMATION EN PSEUDO-UNIFORMES")
print("=" * 64)

n = len(X)
U = np.column_stack([stats.rankdata(X[:,k]) / (n+1) for k in range(X.shape[1])])
U_fort = np.asfortranarray(U)
print(f"  U : {U.shape}  range [{U.min():.4f}, {U.max():.4f}]")

# ─────────────────────────────────────────────────────────────
# 6. SÉLECTION DE COPULE  (paires significatives uniquement)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 6 — SÉLECTION DE COPULE (paires significatives)")
print("=" * 64)

fam_set = [pv.BicopFamily.gumbel, pv.BicopFamily.clayton,
           pv.BicopFamily.frank,  pv.BicopFamily.gaussian,
           pv.BicopFamily.indep]

cop_results = []
sig_pairs   = [(i,j) for i,j in pairs
               if dep_df[dep_df['Paire']==f"{labels[i]} ↔ {labels[j]}"]['Sig.'].values[0]=='✓']

for (i,j) in sig_pairs:
    plabel   = f"{labels[i]} ↔ {labels[j]}"
    u_pair   = np.asfortranarray(U[:,[i,j]])
    ctrl     = pv.FitControlsBicop(family_set=fam_set)
    cop      = pv.Bicop(); cop.select(u_pair, ctrl)

    scores = {}
    for fam in fam_set[:-1]:
        try:
            c = pv.Bicop(family=fam); c.fit(u_pair)
            ll  = c.loglik(u_pair)
            scores[fam.name] = -2*ll + 2*len(c.parameters.flatten())
        except: pass

    pv_val = round(float(cop.parameters.flatten()[0]),4) \
             if len(cop.parameters.flatten()) > 0 else 'N/A'
    cop_results.append({'Paire': plabel, 'Copule': cop.family.name,
                        'Paramètre': pv_val, 'τ copule': round(cop.tau,4),
                        'Log-Lik': round(cop.loglik(u_pair),2)})

    best_aic_fam = min(scores, key=scores.get) if scores else cop.family.name
    print(f"  {plabel}")
    print(f"    → {cop.family.name:10s}  τ={cop.tau:.3f}  " +
          " | ".join(f"{f}={a:.1f}" for f,a in sorted(scores.items(), key=lambda x: x[1])))

cop_df = pd.DataFrame(cop_results)

# ─────────────────────────────────────────────────────────────
# 7. VINE COPULE (8 garanties)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 7 — VINE COPULE (modèle joint 8 garanties)")
print("=" * 64)

ctrl_vine = pv.FitControlsVinecop(family_set=fam_set, trunc_lvl=2)
vine = pv.Vinecop(U.shape[1])
vine.select(U_fort, ctrl_vine)
print(f"  Log-vraisemblance : {vine.loglik(U_fort):.2f}")

# ─────────────────────────────────────────────────────────────
# 8. MONTE CARLO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 8 — MONTE CARLO (N=100 000 scénarios)")
print("=" * 64)

N_SIM  = 100_000
U_sim  = vine.simulate(N_SIM, seeds=[42])
X_sim  = np.zeros_like(U_sim)

for k, c in enumerate(cols_adj):
    dn, par = best_fits[c]['dist'], best_fits[c]['params']
    X_sim[:,k] = DISTS[dn].ppf(np.clip(U_sim[:,k], 1e-6, 1-1e-6), *par)

total = X_sim.sum(axis=1)
print(f"  Charge totale mensuelle  — moyenne : {total.mean()/1e6:,.2f} M€  "
      f"écart-type : {total.std()/1e6:,.2f} M€")

# ─────────────────────────────────────────────────────────────
# 9. VaR & TVaR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 9 — BESOINS EN FONDS PROPRES")
print("=" * 64)

levels      = [0.90, 0.95, 0.99, 0.995, 0.999]
cap_results = []
for alpha in levels:
    var   = np.quantile(total, alpha)
    tvar  = total[total >= var].mean()
    indep = sum(np.quantile(X_sim[:,k], alpha) for k in range(X_sim.shape[1]))
    div   = (indep - var) / indep * 100
    cap_results.append({'Seuil': f"{alpha*100:.1f}%",
                        'VaR (M€)'     : round(var/1e6, 2),
                        'TVaR (M€)'    : round(tvar/1e6, 2),
                        'VaR indép.(M€)': round(indep/1e6, 2),
                        'Gain diversif.': f"{div:.1f}%"})

cap_df = pd.DataFrame(cap_results)
print("\n" + cap_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 10. VISUALISATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ÉTAPE 10 — VISUALISATIONS")
print("=" * 64)

DARK  = '#0f1117'; PANEL = '#1a1d27'; TEXT = '#e0e0e0'; GRID = '#2a2d3a'
ACC8  = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c','#e91e8c']

fig = plt.figure(figsize=(24, 28))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0 — séries temporelles
ax_ts = fig.add_subplot(gs[0, :])
ax_ts.set_facecolor(PANEL)
for k, (c, label, color) in enumerate(zip(cols_adj, labels, ACC8)):
    ax_ts.plot(df['Mois'], df[c]/1e6, color=color, lw=1.2, alpha=0.8, label=label)
ax_ts.set_title('Charges sinistres mensuelles — 8 garanties — Assureur français (1992–2006)',
                color=TEXT, fontsize=12, fontweight='bold')
ax_ts.set_ylabel('Millions €', color=TEXT)
ax_ts.tick_params(colors=TEXT)
ax_ts.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, ncol=8, loc='upper left')
for sp in ax_ts.spines.values(): sp.set_color(GRID)

# Row 1 — 3 histogrammes marginaux (incendie, tempête, gros incendies)
for idx, k in enumerate([0, 1, 6]):
    ax = fig.add_subplot(gs[1, idx]); ax.set_facecolor(PANEL)
    c  = cols_adj[k]; label = labels[k]
    x  = df[c].values
    ax.hist(x/1e6, bins=30, density=True, color=ACC8[k], alpha=0.6, edgecolor='none')
    xr  = np.linspace(x.min(), np.percentile(x, 99), 300)
    dn  = best_fits[c]['dist']; par = best_fits[c]['params']
    ax.plot(xr/1e6, DISTS[dn].pdf(xr, *par)*1e6, color='white', lw=2, label=dn)
    ax.set_title(f'{label}\n({dn})', color=TEXT, fontsize=10, fontweight='bold')
    ax.set_xlabel('M€ / mois', color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
    for sp in ax.spines.values(): sp.set_color(GRID)

# Row 2 — heatmap Kendall 8×8
ax_h = fig.add_subplot(gs[2, :2]); ax_h.set_facecolor(PANEL)
tau_mat = np.eye(len(labels))
for (i,j), plabel in zip(pairs, pair_labels):
    row = dep_df[dep_df['Paire']==plabel]
    if len(row):
        v = float(row['Kendall τ'].values[0])
        tau_mat[i,j] = tau_mat[j,i] = v
im = ax_h.imshow(tau_mat, cmap='RdYlGn', vmin=-0.2, vmax=0.6, aspect='auto')
short = ['Incend.','Tempête','Dommag.','Vol','RC','Autres','Gros Inc.','Gros Dam.']
ax_h.set_xticks(range(8)); ax_h.set_xticklabels(short, color=TEXT, fontsize=7, rotation=35)
ax_h.set_yticks(range(8)); ax_h.set_yticklabels(short, color=TEXT, fontsize=7)
for ii in range(8):
    for jj in range(8):
        ax_h.text(jj, ii, f'{tau_mat[ii,jj]:.2f}', ha='center', va='center',
                  color='black', fontsize=7, fontweight='bold')
ax_h.set_title('Matrice Kendall τ — 8 garanties  (28 paires)', color=TEXT,
               fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax_h).ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
for sp in ax_h.spines.values(): sp.set_color(GRID)

# Row 2 right — copules bar
ax_c = fig.add_subplot(gs[2, 2]); ax_c.set_facecolor(PANEL)
fmap = {'gumbel':'#e74c3c','clayton':'#3498db','frank':'#2ecc71',
        'gaussian':'#9b59b6','indep':'#95a5a6'}
short_p = [p.replace('Incendie (total)','Incend.').replace('Dommages (total)','Dommag.')
            .replace('Tempête','Temp.').replace('Gros Incendies','GrosInc.')
            .replace('Gros Dommages','GrosDam.').replace('RC (TPL)','RC') for p in cop_df['Paire']]
bc   = [fmap.get(f.lower(),'#aaa') for f in cop_df['Copule']]
bars = ax_c.barh(range(len(cop_df)), cop_df['Log-Lik'], color=bc, edgecolor='none', height=0.6)
ax_c.set_yticks(range(len(cop_df)))
ax_c.set_yticklabels(short_p, color=TEXT, fontsize=6)
for bar, fname in zip(bars, cop_df['Copule']):
    w = bar.get_width()
    ax_c.text(max(w*0.5, 0.3), bar.get_y()+bar.get_height()/2,
              fname.capitalize(), ha='center', va='center', color='white', fontsize=7, fontweight='bold')
ax_c.set_title('Copule sélectionnée\npar paire (Log-Lik)', color=TEXT, fontsize=10, fontweight='bold')
ax_c.set_xlabel('Log-Lik', color=TEXT, fontsize=9)
ax_c.tick_params(colors=TEXT, labelsize=6)
for sp in ax_c.spines.values(): sp.set_color(GRID)

# Row 3 — distribution agrégée + VaR/TVaR bar
ax_l = fig.add_subplot(gs[3, :2]); ax_l.set_facecolor(PANEL)
pct  = np.percentile(total, 99.5)
ax_l.hist(total[total<=pct]/1e6, bins=120, density=True,
          color='#6c8ebf', alpha=0.7, edgecolor='none')
for alpha_v, cv, res in zip(levels,
    ['#f39c12','#e67e22','#e74c3c','#c0392b','#922b21'], cap_results):
    v = res['VaR (M€)']
    if v <= pct/1e6:
        ax_l.axvline(v, color=cv, lw=2, ls='--',
                     label=f"VaR {res['Seuil']} = {v:.1f} M€")
ax_l.set_title('Distribution charge agrégée mensuelle (8 garanties) — Seuils VaR',
               color=TEXT, fontsize=11, fontweight='bold')
ax_l.set_xlabel('Charge totale mensuelle (M€)', color=TEXT)
ax_l.set_ylabel('Densité', color=TEXT)
ax_l.tick_params(colors=TEXT)
ax_l.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, ncol=2, loc='upper right')
for sp in ax_l.spines.values(): sp.set_color(GRID)

ax_k = fig.add_subplot(gs[3, 2]); ax_k.set_facecolor(PANEL)
xp = np.arange(len(levels)); w = 0.25
ax_k.bar(xp-w, [r['VaR (M€)']      for r in cap_results], w, color='#3498db', label='VaR (copule)', alpha=0.9)
ax_k.bar(xp,   [r['TVaR (M€)']     for r in cap_results], w, color='#e74c3c', label='TVaR (copule)',alpha=0.9)
ax_k.bar(xp+w, [r['VaR indép.(M€)']for r in cap_results], w, color='#95a5a6', label='VaR (indép.)', alpha=0.9)
ax_k.set_xticks(xp)
ax_k.set_xticklabels([r['Seuil'] for r in cap_results], color=TEXT, fontsize=8)
ax_k.set_title('VaR vs TVaR vs\nIndépendance', color=TEXT, fontsize=10, fontweight='bold')
ax_k.set_ylabel('M€ / mois', color=TEXT, fontsize=9)
ax_k.tick_params(colors=TEXT, labelsize=8)
ax_k.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
for sp in ax_k.spines.values(): sp.set_color(GRID)

fig.suptitle('Copules & Dépendances — Méthodologie Cadoux (2004) — Version 2\n'
             '8 garanties | 180 mois | freclaimset3multi9207 + fire9207 + dam9207',
             color=TEXT, fontsize=13, fontweight='bold', y=0.98)

plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"  Graphique : {OUT_PNG}")

# ─── RÉSUMÉ ──────────────────────────────────────────────────
print("\n" + "=" * 64)
print("RÉSUMÉ FINAL")
print("=" * 64)
print(f"\n  Mois : {len(df)}  |  Garanties : {len(cols)}  |  Paires : {len(pairs)}")
print(f"  Paires significatives : {len(sig_df)}")
print("\n── Lois marginales ──")
for c, l in zip(cols_adj, labels):
    print(f"  {l:22s} → {best_fits[c]['dist']}")
print("\n── Dépendances significatives ──")
print(sig_df.to_string(index=False))
print("\n── Copules sélectionnées ──")
print(cop_df[['Paire','Copule','Paramètre','τ copule']].to_string(index=False))
print("\n── Besoins en fonds propres ──")
print(cap_df.to_string(index=False))
print("\n✓ Pipeline Cadoux v2 complet.")
