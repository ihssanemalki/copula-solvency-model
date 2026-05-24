"""
Visualisations individuelles 
=========================================================
Genere chaque graphique separement dans le meme dossier.
Fichier requis : monthly_claims_combined.csv
"""
import sys, io, os, itertools, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyvinecopulib as pv

# -- Chemins --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, r"C:\Users\HP\Downloads\monthly_claims_combined.csv")

# -- Couleurs & style -----------------------------------------
DARK  = '#0f1117'
PANEL = '#1a1d27'
TEXT  = '#e0e0e0'
GRID  = '#2a2d3a'
ACC8  = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c','#e91e8c']

np.random.seed(42)

# ============================================================
# PREPARATION DES DONNEES (commune a toutes les figures)
# ============================================================
print("Chargement et preparation des donnees...")

df = pd.read_csv(CSV_PATH)
df['Mois'] = pd.to_datetime(df['Mois'])
df = df.sort_values('Mois').reset_index(drop=True)

GARANTIES = {
    'Fire_Claim'      : 'Incendie (total)',
    'HSS_Claim'       : 'Tempete',
    'Damage_Claim'    : 'Dommages (total)',
    'Thief_Claim'     : 'Vol',
    'TPL_Claim'       : 'RC (TPL)',
    'Other_Claim'     : 'Autres',
    'LargeFire_Claim' : 'Gros Incendies',
    'LargeDam_Claim'  : 'Gros Dommages',
}
cols   = list(GARANTIES.keys())
labels = list(GARANTIES.values())
SHORT  = ['Incend.','Tempete','Dommag.','Vol','RC','Autres','Gros Inc.','Gros Dam.']

# Redressement inflation
base_year = 2006
df['annee'] = df['Mois'].dt.year
for c in cols:
    df[c+'_adj'] = df[c] * (1.02)**(base_year - df['annee'])
cols_adj = [c+'_adj' for c in cols]

# Lois marginales
DISTS = {'Lognormale': stats.lognorm,
         'Weibull'   : stats.weibull_min,
         'Gamma'     : stats.gamma}
best_fits = {}
for c, label in zip(cols_adj, labels):
    x = df[c].values
    res = {}
    for dname, dist in DISTS.items():
        params = dist.fit(x, floc=0)
        ll     = np.sum(dist.logpdf(x, *params))
        res[dname] = {'params': params, 'aic': 2*len(params)-2*ll}
    best = min(res, key=lambda d: res[d]['aic'])
    best_fits[c] = {'dist': best, 'params': res[best]['params']}

# Dependances
X     = df[cols_adj].values
pairs = list(itertools.combinations(range(len(cols_adj)), 2))
pair_labels = [f"{labels[i]} <-> {labels[j]}" for i,j in pairs]
dep_results = []
for (i,j), plabel in zip(pairs, pair_labels):
    tau, p_tau = kendalltau(X[:,i], X[:,j])
    rho, _     = spearmanr(X[:,i], X[:,j])
    dep_results.append({'Paire': plabel, 'Kendall tau': round(tau,4),
                        'p-val': round(p_tau,4), 'Spearman rho': round(rho,4),
                        'Sig.': '[OK]' if p_tau < 0.05 else '[NO]'})
dep_df = pd.DataFrame(dep_results)
sig_df = dep_df[dep_df['Sig.']=='[OK]'].sort_values('Kendall tau', ascending=False)

# Pseudo-uniformes
n = len(X)
U = np.column_stack([stats.rankdata(X[:,k])/(n+1) for k in range(X.shape[1])])
U_fort = np.asfortranarray(U)

# Copules
fam_set = [pv.BicopFamily.gumbel, pv.BicopFamily.clayton,
           pv.BicopFamily.frank,  pv.BicopFamily.gaussian,
           pv.BicopFamily.indep]
cop_results = []
sig_pairs = [(i,j) for i,j in pairs
             if dep_df[dep_df['Paire']==f"{labels[i]} <-> {labels[j]}"]['Sig.'].values[0]=='[OK]']
for (i,j) in sig_pairs:
    plabel = f"{labels[i]} <-> {labels[j]}"
    u_pair = np.asfortranarray(U[:,[i,j]])
    cop    = pv.Bicop(); cop.select(u_pair, pv.FitControlsBicop(family_set=fam_set))
    pv_val = round(float(cop.parameters.flatten()[0]),4) \
             if len(cop.parameters.flatten())>0 else 'N/A'
    cop_results.append({'Paire': plabel, 'Copule': cop.family.name,
                        'Parametre': pv_val, 'tau copule': round(cop.tau,4),
                        'Log-Lik': round(cop.loglik(u_pair),2)})
cop_df = pd.DataFrame(cop_results)

# Vine + Monte Carlo
ctrl_vine = pv.FitControlsVinecop(family_set=fam_set, trunc_lvl=2)
vine = pv.Vinecop(U.shape[1]); vine.select(U_fort, ctrl_vine)
N_SIM = 100_000
U_sim = vine.simulate(N_SIM, seeds=[42])
X_sim = np.zeros_like(U_sim)
for k, c in enumerate(cols_adj):
    dn, par = best_fits[c]['dist'], best_fits[c]['params']
    X_sim[:,k] = DISTS[dn].ppf(np.clip(U_sim[:,k],1e-6,1-1e-6), *par)
total = X_sim.sum(axis=1)

# Capital
levels = [0.90, 0.95, 0.99, 0.995, 0.999]
cap_results = []
for alpha in levels:
    var   = np.quantile(total, alpha)
    tvar  = total[total >= var].mean()
    indep = sum(np.quantile(X_sim[:,k], alpha) for k in range(X_sim.shape[1]))
    div   = (indep - var)/indep*100
    cap_results.append({'Seuil': f"{alpha*100:.1f}%",
                        'VaR': var, 'TVaR': tvar,
                        'VaR_indep': indep, 'Div': div})

print("Donnees pretes. Generation des figures...\n")

# ============================================================
# FIGURE 1 -- Series temporelles mensuelles
# ============================================================
def fig1_series_temporelles():
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)

    for k, (c, label, color) in enumerate(zip(cols_adj, labels, ACC8)):
        ax.plot(df['Mois'], df[c]/1e6, color=color, lw=1.3,
                alpha=0.85, label=label)

    ax.set_title('Charges sinistres mensuelles par garantie\nAssureur francais non-vie -- 1992--2006',
                 color=TEXT, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Millions EUR', color=TEXT, fontsize=11)
    ax.set_xlabel('Date', color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
              ncol=4, loc='upper right', framealpha=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(axis='y', color=GRID, alpha=0.4, lw=0.5)

    path = os.path.join(SCRIPT_DIR, 'fig1_series_temporelles.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig1_series_temporelles.png")

# ============================================================
# FIGURE 2 -- Distributions marginales (8 histogrammes)
# ============================================================
def fig2_marginales():
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor(DARK)
    axes = axes.flatten()

    for k, (c, label, color) in enumerate(zip(cols_adj, labels, ACC8)):
        ax = axes[k]; ax.set_facecolor(PANEL)
        x  = df[c].values
        ax.hist(x/1e6, bins=35, density=True, color=color,
                alpha=0.65, edgecolor='none')
        xr  = np.linspace(x.min(), np.percentile(x,99), 300)
        dn  = best_fits[c]['dist']; par = best_fits[c]['params']
        ax.plot(xr/1e6, DISTS[dn].pdf(xr,*par)*1e6,
                color='white', lw=2.5, label=dn)
        ax.set_title(f'{label}\n({dn})', color=TEXT,
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('MEUR / mois', color=TEXT, fontsize=9)
        ax.set_ylabel('Densite', color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
        for sp in ax.spines.values(): sp.set_color(GRID)

    fig.suptitle('Ajustement des lois marginales -- 8 garanties',
                 color=TEXT, fontsize=14, fontweight='bold', y=1.01)
    path = os.path.join(SCRIPT_DIR, 'fig2_marginales.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig2_marginales.png")

# ============================================================
# FIGURE 3 -- Matrice Kendall tau (heatmap 8x8)
# ============================================================
def fig3_heatmap_kendall():
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)

    tau_mat = np.eye(len(labels))
    for (i,j), plabel in zip(pairs, pair_labels):
        row = dep_df[dep_df['Paire']==plabel]
        if len(row):
            v = float(row['Kendall tau'].values[0])
            tau_mat[i,j] = tau_mat[j,i] = v

    im = ax.imshow(tau_mat, cmap='RdYlGn', vmin=-0.2, vmax=0.6, aspect='auto')
    ax.set_xticks(range(8))
    ax.set_xticklabels(SHORT, color=TEXT, fontsize=9, rotation=35, ha='right')
    ax.set_yticks(range(8))
    ax.set_yticklabels(SHORT, color=TEXT, fontsize=9)

    for ii in range(8):
        for jj in range(8):
            v    = tau_mat[ii,jj]
            bold = 'bold' if ii!=jj and abs(v)>0.10 else 'normal'
            ax.text(jj, ii, f'{v:.2f}', ha='center', va='center',
                    color='black', fontsize=9, fontweight=bold)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
    ax.set_title('Matrice Kendall tau -- 8 garanties (28 paires)\n'
                 'Valeurs en gras = dependance significative (|tau| > 0.10)',
                 color=TEXT, fontsize=12, fontweight='bold', pad=12)
    for sp in ax.spines.values(): sp.set_color(GRID)

    path = os.path.join(SCRIPT_DIR, 'fig3_heatmap_kendall.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig3_heatmap_kendall.png")

# ============================================================
# FIGURE 4 -- Scatter plots copule (uniformes empiriques)
#            3 paires les plus fortes
# ============================================================
def fig4_scatter_copules():
    top3 = sig_df.head(3)['Paire'].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK)
    pair_colors = ['#e67e22','#9b59b6','#1abc9c']

    for ax, plabel, color in zip(axes, top3, pair_colors):
        ax.set_facecolor(PANEL)
        idx_i = labels.index(plabel.split(' <-> ')[0])
        idx_j = labels.index(plabel.split(' <-> ')[1])
        tau_v = dep_df[dep_df['Paire']==plabel]['Kendall tau'].values[0]
        cop_v = cop_df[cop_df['Paire']==plabel]['Copule'].values[0] \
                if plabel in cop_df['Paire'].values else 'N/A'

        ax.scatter(U[:,idx_i], U[:,idx_j], alpha=0.4, s=18, color=color)
        ax.set_title(f'{plabel}\ntau = {tau_v:.3f}  |  Copule : {cop_v.capitalize()}',
                     color=TEXT, fontsize=10, fontweight='bold')
        ax.set_xlabel(f'U({SHORT[idx_i]})', color=TEXT, fontsize=9)
        ax.set_ylabel(f'U({SHORT[idx_j]})', color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        # Diagonal reference
        ax.plot([0,1],[0,1], color='white', lw=0.8, alpha=0.3, linestyle='--')
        for sp in ax.spines.values(): sp.set_color(GRID)

    fig.suptitle('Espace des copules -- 3 paires avec la dependance la plus forte',
                 color=TEXT, fontsize=13, fontweight='bold', y=1.02)
    path = os.path.join(SCRIPT_DIR, 'fig4_scatter_copules.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig4_scatter_copules.png")

# ============================================================
# FIGURE 5 -- Copule selectionnee par paire (barres horizontales)
# ============================================================
def fig5_copules_bar():
    fmap = {'gumbel':'#e74c3c','clayton':'#3498db','frank':'#2ecc71',
            'gaussian':'#9b59b6','indep':'#95a5a6'}

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)

    short_p = [p.replace('Incendie (total)','Incend.')
                .replace('Dommages (total)','Dommag.')
                .replace('Gros Incendies','GrosInc.')
                .replace('Gros Dommages','GrosDam.')
                .replace('RC (TPL)','RC') for p in cop_df['Paire']]
    bc   = [fmap.get(f.lower(),'#aaa') for f in cop_df['Copule']]
    ypos = range(len(cop_df))
    bars = ax.barh(list(ypos), cop_df['Log-Lik'], color=bc,
                   edgecolor='none', height=0.65)

    for bar, fname, ll in zip(bars, cop_df['Copule'], cop_df['Log-Lik']):
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                f'{fname.capitalize()}  ({ll:.1f})',
                va='center', color='white', fontsize=9)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(short_p, color=TEXT, fontsize=8)
    ax.set_xlabel('Log-vraisemblance', color=TEXT, fontsize=10)
    ax.set_title('Copule selectionnee par paire significative\n(critere AIC, methode CML)',
                 color=TEXT, fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors=TEXT)
    ax.grid(axis='x', color=GRID, alpha=0.4, lw=0.5)
    for sp in ax.spines.values(): sp.set_color(GRID)

    # Legende couleurs
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k.capitalize())
                       for k,v in fmap.items()]
    ax.legend(handles=legend_elements, fontsize=9, facecolor=PANEL,
              labelcolor=TEXT, loc='lower right')

    path = os.path.join(SCRIPT_DIR, 'fig5_copules_bar.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig5_copules_bar.png")

# ============================================================
# FIGURE 6 -- Distribution perte agregee + seuils VaR
# ============================================================
def fig6_distribution_agregee():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)

    pct = np.percentile(total, 99.5)
    ax.hist(total[total<=pct]/1e6, bins=150, density=True,
            color='#6c8ebf', alpha=0.75, edgecolor='none',
            label='Scenarios simules (100 000)')

    cmap_v = ['#f39c12','#e67e22','#e74c3c','#c0392b','#922b21']
    for cv, res in zip(cmap_v, cap_results):
        v = res['VaR']
        if v/1e6 <= pct/1e6:
            ax.axvline(v/1e6, color=cv, lw=2, linestyle='--',
                       label=f"VaR {res['Seuil']} = {v/1e6:.1f} MEUR")

    # Zone de queue
    x99 = cap_results[2]['VaR']/1e6
    ax.axvspan(x99, pct/1e6, alpha=0.08, color='#e74c3c',
               label='Queue (>VaR 99%)')

    ax.set_title('Distribution de la charge agregee mensuelle (8 garanties)\n'
                 'Simulation Monte Carlo -- Vine Copule',
                 color=TEXT, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Charge totale mensuelle (MEUR)', color=TEXT, fontsize=11)
    ax.set_ylabel('Densite', color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT)
    ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT,
              ncol=2, loc='upper right', framealpha=0.85)
    ax.grid(axis='y', color=GRID, alpha=0.3, lw=0.5)
    for sp in ax.spines.values(): sp.set_color(GRID)

    path = os.path.join(SCRIPT_DIR, 'fig6_distribution_agregee.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig6_distribution_agregee.png")

# ============================================================
# FIGURE 7 -- VaR vs TVaR vs Independance (barres groupees)
# ============================================================
def fig7_capital_var_tvar():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)

    xp = np.arange(len(levels)); w = 0.25
    ax.bar(xp-w, [r['VaR']/1e6      for r in cap_results],
           w, color='#3498db', label='VaR (copule)',  alpha=0.9)
    ax.bar(xp,   [r['TVaR']/1e6     for r in cap_results],
           w, color='#e74c3c', label='TVaR (copule)', alpha=0.9)
    ax.bar(xp+w, [r['VaR_indep']/1e6 for r in cap_results],
           w, color='#95a5a6', label='VaR (independance)', alpha=0.9)

    # Annotations % gain
    for i, res in enumerate(cap_results):
        ax.text(i-w/2, res['VaR']/1e6 + 2,
                f"-{res['Div']:.0f}%", ha='center',
                color='#3498db', fontsize=8, fontweight='bold')

    ax.set_xticks(xp)
    ax.set_xticklabels([r['Seuil'] for r in cap_results],
                       color=TEXT, fontsize=10)
    ax.set_ylabel('MEUR / mois', color=TEXT, fontsize=11)
    ax.set_title('Besoins en fonds propres -- VaR vs TVaR vs Hypothese d\'independance\n'
                 'Les % indiquent le gain de diversification de la copule vs independance',
                 color=TEXT, fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors=TEXT)
    ax.legend(fontsize=10, facecolor=PANEL, labelcolor=TEXT)
    ax.grid(axis='y', color=GRID, alpha=0.3, lw=0.5)
    for sp in ax.spines.values(): sp.set_color(GRID)

    path = os.path.join(SCRIPT_DIR, 'fig7_capital_var_tvar.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig7_capital_var_tvar.png")

# ============================================================
# FIGURE 8 -- Tableau recapitulatif des resultats
# ============================================================
def fig8_tableau_recap():
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(DARK)

    def make_table(ax, data, col_labels, row_labels, title, col_colors):
        ax.set_facecolor(DARK); ax.axis('off')
        table = ax.table(
            cellText=data,
            colLabels=col_labels,
            rowLabels=row_labels,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor(PANEL if r > 0 else '#2a2d3a')
            cell.set_text_props(color=TEXT)
            cell.set_edgecolor(GRID)
        ax.set_title(title, color=TEXT, fontsize=11,
                     fontweight='bold', pad=15)

    # Tableau 1 -- Lois marginales
    data1 = [[best_fits[c]['dist'],
              f"{stats.lognorm.fit(df[c].values, floc=0)[0]:.2f}" if best_fits[c]['dist']=='Lognormale' else '--']
             for c in cols_adj]
    make_table(axes[0], data1,
               ['Loi retenue','Param. sigma'],
               [l[:15] for l in labels],
               'Lois marginales', None)

    # Tableau 2 -- Top dependances
    top8 = sig_df.head(8)
    data2 = [[f"{row['Kendall tau']:.3f}", f"{row['p-val']:.4f}",
              f"{row['Spearman rho']:.3f}"]
             for _, row in top8.iterrows()]
    make_table(axes[1], data2,
               ['Kendall tau','p-val','Spearman rho'],
               [p[:22] for p in top8['Paire']],
               'Top 8 dependances', None)

    # Tableau 3 -- Capital
    data3 = [[f"{r['VaR']/1e6:.1f}",
              f"{r['TVaR']/1e6:.1f}",
              f"{r['VaR_indep']/1e6:.1f}",
              f"{r['Div']:.1f}%"]
             for r in cap_results]
    make_table(axes[2], data3,
               ['VaR','TVaR','VaR indep.','Gain %'],
               [r['Seuil'] for r in cap_results],
               'Capital (MEUR/mois)', None)

    fig.suptitle('Resume des resultats ',
                 color=TEXT, fontsize=14, fontweight='bold', y=1.02)

    path = os.path.join(SCRIPT_DIR, 'fig8_tableau_recap.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  [OK] fig8_tableau_recap.png")

# ============================================================
# LANCEMENT DE TOUTES LES FIGURES
# ============================================================
print("=" * 50)
print("Generation des figures individuelles")
print("=" * 50)

fig1_series_temporelles()
fig2_marginales()
fig3_heatmap_kendall()
fig4_scatter_copules()
fig5_copules_bar()
fig6_distribution_agregee()
fig7_capital_var_tvar()
fig8_tableau_recap()

print("\n" + "=" * 50)
print(f"8 figures sauvegardees dans :")
print(f"  {SCRIPT_DIR}")
print("=" * 50)