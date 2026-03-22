# 📊 Prédiction du Besoin en Fonds de Roulement (BFR) des Entreprises Marocaines

**Objectif :** Prédire le BFR prévisionnel en Dirhams (DH) à partir de données bilantaires simulées (BAM + HCP)  
**Méthode :** Régression supervisée (Ridge, Random Forest, Gradient Boosting, XGBoost)  
**Application :** Gestion de trésorerie · Conseil en optimisation du BFR · Crédits de fonctionnement bancaires

---
> **Sources simulées :** Bilans sectoriels Bank Al-Maghrib (BAM) + Données HCP (Haut-Commissariat au Plan)  
> **Secteurs couverts :** BTP, Industrie, Commerce, Agroalimentaire, Services, Tourisme, Transport

## 1. 📦 Installation & Imports

# Installation des librairies supplémentaires
!pip install xgboost shap --quiet

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

# Style global
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#2e3250',
    'axes.labelcolor': '#c9d1e0',
    'axes.titlecolor': '#ffffff',
    'xtick.color': '#8890a6',
    'ytick.color': '#8890a6',
    'text.color': '#c9d1e0',
    'grid.color': '#2e3250',
    'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

PALETTE = ['#4e8ef7', '#f7a04e', '#4ef7a0', '#f74e8e', '#a04ef7', '#f7f74e', '#4ef7f7']
print('✅ Environnement prêt.')

## 2. 🏭 Simulation des Données (Bilans BAM + HCP)
Données simulées calées sur les ratios sectoriels réels publiés par la BAM et le HCP Maroc.

np.random.seed(42)
N = 1200  # Nombre d'entreprises simulées

# ── Secteurs & paramètres BAM ──────────────────────────────────────────────
SECTEURS = {
    'BTP':            {'ca_moy': 45e6,  'ca_std': 20e6,  'bfr_ratio': 0.18, 'n': 180},
    'Industrie':      {'ca_moy': 80e6,  'ca_std': 35e6,  'bfr_ratio': 0.14, 'n': 220},
    'Commerce':       {'ca_moy': 30e6,  'ca_std': 12e6,  'bfr_ratio': 0.08, 'n': 250},
    'Agroalimentaire':{'ca_moy': 55e6,  'ca_std': 25e6,  'bfr_ratio': 0.10, 'n': 180},
    'Services':       {'ca_moy': 20e6,  'ca_std': 10e6,  'bfr_ratio': 0.06, 'n': 200},
    'Tourisme':       {'ca_moy': 25e6,  'ca_std': 15e6,  'bfr_ratio': 0.12, 'n': 90},
    'Transport':      {'ca_moy': 35e6,  'ca_std': 18e6,  'bfr_ratio': 0.09, 'n': 80},
}

TAILLES    = ['TPE', 'PME', 'ETI', 'GE']
REGIONS    = ['Casablanca-Settat', 'Rabat-Salé', 'Marrakech-Safi',
              'Fès-Meknès', 'Tanger-Tetouan', 'Souss-Massa', 'Oriental']
ANNEES     = list(range(2018, 2024))

rows = []
for secteur, params in SECTEURS.items():
    for _ in range(params['n']):
        ca          = max(500_000, np.random.normal(params['ca_moy'], params['ca_std']))
        taille      = np.random.choice(TAILLES, p=[0.40, 0.35, 0.15, 0.10])
        annee       = np.random.choice(ANNEES)
        region      = np.random.choice(REGIONS)

        # Postes bilantaires (ratios BAM)
        stocks      = ca * np.random.uniform(0.06, 0.20)
        creances_cl = ca * np.random.uniform(0.08, 0.25)
        dettes_four = ca * np.random.uniform(0.05, 0.18)
        dettes_fisc = ca * np.random.uniform(0.01, 0.05)
        tresorerie  = ca * np.random.uniform(-0.03, 0.10)

        # Ratios de rotation (jours)
        delai_client   = (creances_cl / ca) * 365
        delai_four     = (dettes_four / ca) * 365
        rotation_stock = ca / max(stocks, 1)

        # Ratios financiers HCP
        ratio_endet    = np.random.uniform(0.2, 0.8)
        marge_brute    = np.random.uniform(0.08, 0.35)
        ratio_liquidite= np.random.uniform(0.5, 2.5)
        croissance_ca  = np.random.uniform(-0.05, 0.20)

        # BFR = Stocks + Créances clients - Dettes fournisseurs - Dettes fiscales
        bfr_theorique  = stocks + creances_cl - dettes_four - dettes_fisc
        # Ajout d'un bruit réaliste
        bruit          = np.random.normal(0, bfr_theorique * 0.05)
        bfr            = max(0, bfr_theorique + bruit)

        # Besoin crédit de fonctionnement recommandé (80% du BFR non couvert)
        couverture_fdr = bfr * np.random.uniform(0.3, 0.9)
        credit_fonct   = max(0, (bfr - couverture_fdr) * 0.80)

        rows.append({
            'secteur': secteur, 'taille': taille, 'region': region, 'annee': annee,
            'ca': ca, 'stocks': stocks, 'creances_clients': creances_cl,
            'dettes_fournisseurs': dettes_four, 'dettes_fiscales': dettes_fisc,
            'tresorerie': tresorerie, 'delai_client_j': delai_client,
            'delai_fournisseur_j': delai_four, 'rotation_stock': rotation_stock,
            'ratio_endettement': ratio_endet, 'marge_brute': marge_brute,
            'ratio_liquidite': ratio_liquidite, 'croissance_ca': croissance_ca,
            'couverture_fdr': couverture_fdr, 'credit_fonctionnement': credit_fonct,
            'bfr': bfr
        })

df = pd.DataFrame(rows)
print(f'✅ Dataset généré : {df.shape[0]} entreprises × {df.shape[1]} variables')
print(f'   BFR moyen : {df.bfr.mean()/1e6:.2f} M DH  |  médiane : {df.bfr.median()/1e6:.2f} M DH')
df.head()

## 3. 🔍 Analyse Exploratoire (EDA)

# ── 3.1 Distribution du BFR ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution du BFR — Vue générale', fontsize=15, color='white', y=1.02)

# Histogramme
axes[0].hist(df['bfr']/1e6, bins=50, color='#4e8ef7', edgecolor='none', alpha=0.85)
axes[0].set_title('Distribution du BFR (M DH)')
axes[0].set_xlabel('BFR (millions DH)')
axes[0].set_ylabel('Fréquence')
axes[0].axvline(df['bfr'].mean()/1e6, color='#f7a04e', lw=2, linestyle='--', label=f'Moyenne : {df.bfr.mean()/1e6:.1f}M')
axes[0].legend()

# BFR par secteur (boxplot)
order_s = df.groupby('secteur')['bfr'].median().sort_values().index
bfr_s = [df[df.secteur==s]['bfr'].values/1e6 for s in order_s]
bp = axes[1].boxplot(bfr_s, patch_artist=True, vert=True, labels=order_s)
for patch, color in zip(bp['boxes'], PALETTE): patch.set_facecolor(color); patch.set_alpha(0.8)
for median in bp['medians']: median.set_color('white'); median.set_linewidth(2)
axes[1].set_title('BFR par secteur (M DH)')
axes[1].set_ylabel('BFR (M DH)')
axes[1].tick_params(axis='x', rotation=35)

# BFR moyen par taille
bfr_taille = df.groupby('taille')['bfr'].mean().reindex(['TPE','PME','ETI','GE'])/1e6
bars = axes[2].bar(bfr_taille.index, bfr_taille.values,
                   color=['#4e8ef7','#4ef7a0','#f7a04e','#f74e8e'], alpha=0.85, edgecolor='none')
for bar, val in zip(bars, bfr_taille.values):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:.1f}M', ha='center', color='white', fontsize=10)
axes[2].set_title('BFR moyen par taille (M DH)')
axes[2].set_ylabel('BFR moyen (M DH)')

plt.tight_layout()
plt.savefig('/content/fig_distribution_bfr.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

# ── 3.2 Corrélations & Heatmap ────────────────────────────────────────────
num_cols = ['ca','stocks','creances_clients','dettes_fournisseurs','dettes_fiscales',
            'tresorerie','delai_client_j','delai_fournisseur_j','rotation_stock',
            'ratio_endettement','marge_brute','ratio_liquidite','croissance_ca','bfr']

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=0.3, linecolor='#0f1117',
            annot=True, fmt='.2f', annot_kws={'size': 8, 'color': 'white'},
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Matrice de Corrélation — Variables bilantaires', fontsize=14, pad=15)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('/content/fig_correlation.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

# ── 3.3 BFR / CA par secteur & évolution temporelle ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Ratio BFR/CA par secteur
df['bfr_sur_ca'] = df['bfr'] / df['ca']
ratio_s = df.groupby('secteur')['bfr_sur_ca'].mean().sort_values(ascending=True)
colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(ratio_s))]
bars2 = axes[0].barh(ratio_s.index, ratio_s.values * 100, color=colors_bar, alpha=0.85, edgecolor='none')
for bar, val in zip(bars2, ratio_s.values):
    axes[0].text(val*100+0.1, bar.get_y()+bar.get_height()/2, f'{val*100:.1f}%', va='center', color='white', fontsize=10)
axes[0].set_title('Ratio BFR/CA moyen par secteur (%)')
axes[0].set_xlabel('BFR / CA (%)')
axes[0].axvline(df['bfr_sur_ca'].mean()*100, color='#f74e8e', lw=1.5, linestyle='--', label='Moyenne globale')
axes[0].legend()

# Évolution BFR moyen par année
bfr_an = df.groupby(['annee','secteur'])['bfr'].mean().reset_index()
for i, sec in enumerate(SECTEURS.keys()):
    sub = bfr_an[bfr_an.secteur == sec]
    axes[1].plot(sub['annee'], sub['bfr']/1e6, marker='o', markersize=5,
                 label=sec, color=PALETTE[i % len(PALETTE)], linewidth=2)
axes[1].set_title('Évolution du BFR moyen par secteur (M DH)')
axes[1].set_xlabel('Année')
axes[1].set_ylabel('BFR moyen (M DH)')
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/fig_ratios.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

## 4. ⚙️ Préparation des Données & Feature Engineering

# ── Feature Engineering ───────────────────────────────────────────────────
df['bfr_exploitation']  = df['stocks'] + df['creances_clients'] - df['dettes_fournisseurs']
df['couverture_stocks'] = df['stocks'] / df['ca']
df['intensite_client']  = df['creances_clients'] / df['ca']
df['levier_four']       = df['dettes_fournisseurs'] / df['ca']
df['nfr_net']           = df['bfr'] / df['ca']   # BFR normalisé
df['log_ca']            = np.log1p(df['ca'])
df['log_bfr']           = np.log1p(df['bfr'])    # target transformée

# Encodage catégoriel
le_secteur = LabelEncoder()
le_taille  = LabelEncoder()
le_region  = LabelEncoder()
df['secteur_enc'] = le_secteur.fit_transform(df['secteur'])
df['taille_enc']  = le_taille.fit_transform(df['taille'])
df['region_enc']  = le_region.fit_transform(df['region'])

# Features finales
FEATURES = [
    'log_ca', 'stocks', 'creances_clients', 'dettes_fournisseurs', 'dettes_fiscales',
    'tresorerie', 'delai_client_j', 'delai_fournisseur_j', 'rotation_stock',
    'ratio_endettement', 'marge_brute', 'ratio_liquidite', 'croissance_ca',
    'bfr_exploitation', 'couverture_stocks', 'intensite_client', 'levier_four',
    'secteur_enc', 'taille_enc', 'region_enc', 'annee'
]
TARGET = 'bfr'

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Train / Test split (80/20) stratifié par secteur
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df['secteur']
)

# Normalisation
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'✅ Features : {len(FEATURES)} | Train : {len(X_train)} | Test : {len(X_test)}')

## 5. 🤖 Modélisation & Comparaison des Modèles

# ── Définition des modèles ─────────────────────────────────────────────────
MODELS = {
    'Ridge Regression':        Ridge(alpha=10),
    'Random Forest':           RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting':       GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'XGBoost':                 xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                 subsample=0.8, colsample_bytree=0.8,
                                                 random_state=42, verbosity=0),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print('🔄 Entraînement des modèles...')
for name, model in MODELS.items():
    # Choix de la matrice (Ridge sur données scalées)
    Xtr = X_train_sc if name == 'Ridge Regression' else X_train.values
    Xte = X_test_sc  if name == 'Ridge Regression' else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Cross-validation R²
    cv_r2 = cross_val_score(model, Xtr, y_train, cv=kf, scoring='r2').mean()

    results[name] = {
        'model': model, 'y_pred': y_pred,
        'MAE (DH)': mae, 'RMSE (DH)': rmse,
        'R²': r2, 'MAPE (%)': mape, 'CV R² (5-fold)': cv_r2
    }
    print(f'  ✓ {name:25s}  R²={r2:.4f}  MAE={mae/1e3:.1f}k DH  MAPE={mape:.1f}%')

print('\n✅ Entraînement terminé.')

# ── Tableau comparatif ─────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ('model', 'y_pred')}
    for name, res in results.items()
}).T

metrics_df['MAE (M DH)']  = metrics_df['MAE (DH)'].astype(float) / 1e6
metrics_df['RMSE (M DH)'] = metrics_df['RMSE (DH)'].astype(float) / 1e6
display_cols = ['R²','CV R² (5-fold)','MAE (M DH)','RMSE (M DH)','MAPE (%)']

print('\n📋 Tableau comparatif des modèles :\n')
styled = metrics_df[display_cols].astype(float).round(4)
print(styled.to_string())

## 6. 📈 Visualisations des Performances

# ── 6.1 Comparaison R² et MAPE ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Performances comparées des modèles de prédiction du BFR', fontsize=14, color='white')

names = list(results.keys())
r2s   = [results[n]['R²'] for n in names]
maes  = [results[n]['MAE (DH)']/1e6 for n in names]
mapes = [results[n]['MAPE (%)'] for n in names]

# R²
bars_r2 = axes[0].bar(names, r2s, color=PALETTE[:4], alpha=0.85, edgecolor='none')
for bar, v in zip(bars_r2, r2s):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.4f}', ha='center', color='white', fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].set_title('R² (Test Set)')
axes[0].tick_params(axis='x', rotation=20)
axes[0].axhline(0.9, color='#f74e8e', lw=1.2, linestyle='--', alpha=0.7, label='Seuil 0.90')
axes[0].legend(fontsize=8)

# MAE
bars_mae = axes[1].bar(names, maes, color=PALETTE[:4], alpha=0.85, edgecolor='none')
for bar, v in zip(bars_mae, maes):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.2f}M', ha='center', color='white', fontsize=9)
axes[1].set_title('MAE (M DH)')
axes[1].set_ylabel('MAE (millions DH)')
axes[1].tick_params(axis='x', rotation=20)

# MAPE
bars_mape = axes[2].bar(names, mapes, color=PALETTE[:4], alpha=0.85, edgecolor='none')
for bar, v in zip(bars_mape, mapes):
    axes[2].text(bar.get_x()+bar.get_width()/2, v+0.05, f'{v:.1f}%', ha='center', color='white', fontsize=9)
axes[2].set_title('MAPE (%)')
axes[2].set_ylabel('Erreur absolue moyenne (%)')
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('/content/fig_performances.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

# ── 6.2 Réel vs Prédit — Meilleur modèle (XGBoost) ────────────────────────
best_model_name = max(results, key=lambda n: results[n]['R²'])
y_pred_best     = results[best_model_name]['y_pred']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Analyse des prédictions — {best_model_name}', fontsize=14, color='white')

# Scatter Réel vs Prédit
vmax = max(y_test.max(), y_pred_best.max()) / 1e6
axes[0].scatter(y_test/1e6, y_pred_best/1e6, alpha=0.4, s=15,
                c=np.abs(y_test - y_pred_best)/y_test, cmap='RdYlGn_r')
axes[0].plot([0, vmax], [0, vmax], 'w--', lw=1.5, label='Prédiction parfaite')
axes[0].set_xlabel('BFR réel (M DH)')
axes[0].set_ylabel('BFR prédit (M DH)')
axes[0].set_title(f'BFR Réel vs Prédit  (R²={results[best_model_name]["R²"]:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Résidus
residus = (y_test - y_pred_best) / 1e6
axes[1].hist(residus, bins=50, color='#4e8ef7', edgecolor='none', alpha=0.85)
axes[1].axvline(0, color='#f7a04e', lw=2, linestyle='--')
axes[1].axvline(residus.mean(), color='#f74e8e', lw=1.5, linestyle=':',
                label=f'Moyenne résidus: {residus.mean():.2f}M')
axes[1].set_xlabel('Résidu (M DH)')
axes[1].set_ylabel('Fréquence')
axes[1].set_title('Distribution des résidus')
axes[1].legend()

plt.tight_layout()
plt.savefig('/content/fig_reel_vs_predit.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

## 7. 🔬 Importance des Variables & Interprétation SHAP

# ── 7.1 Feature Importance (XGBoost natif) ─────────────────────────────────
best_model = results[best_model_name]['model']

if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_fi = [PALETTE[i % len(PALETTE)] for i in range(len(fi))]
    bars_fi = ax.barh(fi.index, fi.values, color=colors_fi, alpha=0.85, edgecolor='none')
    for bar, v in zip(bars_fi, fi.values):
        ax.text(v + 0.001, bar.get_y()+bar.get_height()/2,
                f'{v*100:.1f}%', va='center', color='white', fontsize=9)
    ax.set_title(f'Importance des variables — {best_model_name}', fontsize=13)
    ax.set_xlabel('Importance relative')
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig('/content/fig_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.show()

# ── 7.2 SHAP Values (Explainability) ──────────────────────────────────────
print('🔄 Calcul des SHAP values...')
X_test_df = pd.DataFrame(X_test.values, columns=FEATURES)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_df)

plt.figure(figsize=(12, 7))
plt.title(f'SHAP Summary Plot — {best_model_name}', fontsize=13, color='white', pad=10)
shap.summary_plot(shap_values, X_test_df, plot_type='bar', show=False,
                  color='#4e8ef7', max_display=15)
plt.tight_layout()
plt.savefig('/content/fig_shap.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ SHAP values calculées.')

## 8. 🏦 Analyse Financière : BFR, Crédit de Fonctionnement & Trésorerie

# ── 8.1 Prédictions enrichies avec recommandations bancaires ──────────────
X_test_df_orig = X_test.copy()
y_pred_xgb = results[best_model_name]['y_pred']

resultats = df.loc[X_test.index].copy()
resultats['bfr_predit']   = y_pred_xgb
resultats['erreur_rel_%'] = np.abs(resultats['bfr'] - resultats['bfr_predit']) / resultats['bfr'] * 100
resultats['bfr_vs_ca_%']  = resultats['bfr_predit'] / resultats['ca'] * 100

# Calcul du crédit de fonctionnement recommandé
resultats['besoin_credit_fonct'] = np.maximum(
    0, (resultats['bfr_predit'] - resultats['couverture_fdr']) * 0.80
)

# Diagnostic BFR
def diagnostic_bfr(row):
    ratio = row['bfr_vs_ca_%']
    if ratio < 8:   return '🟢 BFR Optimisé'
    elif ratio < 15: return '🟡 BFR Acceptable'
    elif ratio < 22: return '🟠 BFR Élevé'
    else:            return '🔴 BFR Critique'

resultats['diagnostic'] = resultats.apply(diagnostic_bfr, axis=1)

print('📊 Résumé des prédictions par secteur :\n')
summary = resultats.groupby('secteur').agg(
    BFR_Réel_mDH=('bfr', lambda x: round(x.mean()/1e6, 2)),
    BFR_Prédit_mDH=('bfr_predit', lambda x: round(x.mean()/1e6, 2)),
    Erreur_Moy_pct=('erreur_rel_%', lambda x: round(x.mean(), 2)),
    Crédit_Fonct_mDH=('besoin_credit_fonct', lambda x: round(x.mean()/1e6, 2)),
    N=('bfr', 'count')
)
print(summary.to_string())

# ── 8.2 Dashboard Financier ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Dashboard BFR Prévisionnel — Analyse Financière Sectorielle', 
             fontsize=16, color='white', y=1.01, fontweight='bold')

gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

# ① BFR réel vs prédit par secteur
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(summary))
w = 0.38
ax1.bar(x_pos - w/2, summary['BFR_Réel_mDH'],    width=w, label='Réel',   color='#4e8ef7', alpha=0.85)
ax1.bar(x_pos + w/2, summary['BFR_Prédit_mDH'],   width=w, label='Prédit', color='#4ef7a0', alpha=0.85)
ax1.set_xticks(x_pos); ax1.set_xticklabels(summary.index, rotation=35, ha='right', fontsize=9)
ax1.set_title('BFR Réel vs Prédit (M DH)')
ax1.set_ylabel('M DH'); ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

# ② Crédit de fonctionnement recommandé
ax2 = fig.add_subplot(gs[0, 1])
sorted_credit = resultats.groupby('secteur')['besoin_credit_fonct'].mean().sort_values(ascending=False)/1e6
ax2.bar(sorted_credit.index, sorted_credit.values, color='#f7a04e', alpha=0.85, edgecolor='none')
for i, (sect, val) in enumerate(sorted_credit.items()):
    ax2.text(i, val+0.05, f'{val:.1f}M', ha='center', color='white', fontsize=9)
ax2.set_title('Crédit de Fonctionnement\nRecommandé Moyen (M DH)')
ax2.set_ylabel('M DH'); ax2.tick_params(axis='x', rotation=35); ax2.grid(axis='y', alpha=0.3)

# ③ Distribution des diagnostics
ax3 = fig.add_subplot(gs[0, 2])
diag_counts = resultats['diagnostic'].value_counts()
diag_colors = {'🟢 BFR Optimisé': '#4ef7a0', '🟡 BFR Acceptable': '#f7f74e',
               '🟠 BFR Élevé': '#f7a04e',    '🔴 BFR Critique': '#f74e8e'}
colors_diag = [diag_colors.get(k, '#888') for k in diag_counts.index]
wedges, texts, autotexts = ax3.pie(
    diag_counts.values, labels=None, autopct='%1.1f%%',
    colors=colors_diag, startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2}
)
for at in autotexts: at.set_color('white'); at.set_fontsize(9)
ax3.legend(wedges, diag_counts.index, loc='lower center', bbox_to_anchor=(0.5, -0.22),
           fontsize=8, ncol=2)
ax3.set_title('Diagnostic BFR (Portefeuille)')

# ④ BFR prédit vs délai client (scatter)
ax4 = fig.add_subplot(gs[1, 0])
for i, sec in enumerate(SECTEURS.keys()):
    sub = resultats[resultats.secteur == sec]
    ax4.scatter(sub['delai_client_j'], sub['bfr_predit']/1e6,
                label=sec, color=PALETTE[i % len(PALETTE)], alpha=0.5, s=18)
ax4.set_xlabel('Délai client (jours)')
ax4.set_ylabel('BFR prédit (M DH)')
ax4.set_title('BFR prédit vs Délai client')
ax4.legend(fontsize=7, ncol=2)
ax4.grid(True, alpha=0.2)

# ⑤ Erreur de prédiction par secteur
ax5 = fig.add_subplot(gs[1, 1])
err_s = resultats.groupby('secteur')['erreur_rel_%'].mean().sort_values()
colors_err = ['#4ef7a0' if v < 5 else '#f7a04e' if v < 10 else '#f74e8e' for v in err_s.values]
bars_err = ax5.barh(err_s.index, err_s.values, color=colors_err, alpha=0.85, edgecolor='none')
for bar, v in zip(bars_err, err_s.values):
    ax5.text(v + 0.05, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', color='white', fontsize=9)
ax5.set_title('Erreur Moy. de Prédiction\npar Secteur (%)')
ax5.set_xlabel('MAPE (%)')
ax5.axvline(5, color='white', lw=1, linestyle='--', alpha=0.5, label='Seuil 5%')
ax5.legend(fontsize=8); ax5.grid(axis='x', alpha=0.3)

# ⑥ Carte des besoins en crédit par région
ax6 = fig.add_subplot(gs[1, 2])
credit_reg = resultats.groupby('region')['besoin_credit_fonct'].sum()/1e6
credit_reg = credit_reg.sort_values(ascending=True)
colors_reg = [PALETTE[i % len(PALETTE)] for i in range(len(credit_reg))]
bars_reg = ax6.barh(credit_reg.index, credit_reg.values, color=colors_reg, alpha=0.85, edgecolor='none')
for bar, v in zip(bars_reg, credit_reg.values):
    ax6.text(v + 0.3, bar.get_y()+bar.get_height()/2, f'{v:.0f}M', va='center', color='white', fontsize=9)
ax6.set_title('Besoins Crédit Fonctionnement\npar Région (M DH)')
ax6.set_xlabel('Crédit de fonctionnement (M DH)')
ax6.grid(axis='x', alpha=0.3)

plt.savefig('/content/fig_dashboard_financier.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ Dashboard financier généré.')

## 9. 🔮 Simulateur de Prédiction BFR (Entreprise individuelle)

# ═══════════════════════════════════════════════════════════════════
#  SIMULATEUR — Modifier les paramètres ci-dessous
# ═══════════════════════════════════════════════════════════════════

ENTREPRISE_TEST = {
    # ── Identification ──────────────────────────────────────────────
    'nom':      'SIM AGRO SARL',
    'secteur':  'Agroalimentaire',   # BTP / Industrie / Commerce / Agroalimentaire / Services / Tourisme / Transport
    'taille':   'PME',              # TPE / PME / ETI / GE
    'region':   'Casablanca-Settat',
    'annee':    2024,

    # ── Compte de résultat ───────────────────────────────────────────
    'ca':              42_000_000,   # Chiffre d'affaires (DH)
    'marge_brute':     0.18,         # 18%
    'croissance_ca':   0.08,         # +8%

    # ── Bilan ────────────────────────────────────────────────────────
    'stocks':                  6_800_000,
    'creances_clients':        8_400_000,
    'dettes_fournisseurs':     5_200_000,
    'dettes_fiscales':         1_100_000,
    'tresorerie':              2_300_000,
    'couverture_fdr':          4_000_000,   # FDR couvrant le BFR

    # ── Ratios financiers ────────────────────────────────────────────
    'ratio_endettement':  0.45,
    'ratio_liquidite':    1.35,
}

# ───────────────────────────────────────────────────────────────────
# Calcul des features dérivées
e = ENTREPRISE_TEST
e['delai_client_j']    = (e['creances_clients'] / e['ca']) * 365
e['delai_fournisseur_j'] = (e['dettes_fournisseurs'] / e['ca']) * 365
e['rotation_stock']    = e['ca'] / max(e['stocks'], 1)
e['bfr_exploitation']  = e['stocks'] + e['creances_clients'] - e['dettes_fournisseurs']
e['couverture_stocks'] = e['stocks'] / e['ca']
e['intensite_client']  = e['creances_clients'] / e['ca']
e['levier_four']       = e['dettes_fournisseurs'] / e['ca']
e['log_ca']            = np.log1p(e['ca'])
e['secteur_enc']       = le_secteur.transform([e['secteur']])[0]
e['taille_enc']        = le_taille.transform([e['taille']])[0]
e['region_enc']        = le_region.transform([e['region']])[0]

# Vecteur de prédiction
X_new = pd.DataFrame([[e[f] for f in FEATURES]], columns=FEATURES)
bfr_predit = best_model.predict(X_new)[0]

# BFR comptable (formule théorique)
bfr_comptable = e['stocks'] + e['creances_clients'] - e['dettes_fournisseurs'] - e['dettes_fiscales']

# Crédit de fonctionnement recommandé
credit_recommande = max(0, (bfr_predit - e['couverture_fdr']) * 0.80)

# Diagnostic
ratio_bfr_ca = bfr_predit / e['ca'] * 100
if ratio_bfr_ca < 8:    diag_label = '🟢 BFR Optimisé'
elif ratio_bfr_ca < 15: diag_label = '🟡 BFR Acceptable'
elif ratio_bfr_ca < 22: diag_label = '🟠 BFR Élevé — vigilance requise'
else:                    diag_label = '🔴 BFR Critique — action urgente'

# ── Affichage ─────────────────────────────────────────────────────────────
print('═'*60)
print(f'  RAPPORT BFR PRÉVISIONNEL — {e["nom"]}')
print('═'*60)
print(f'  Secteur     : {e["secteur"]}  |  Taille : {e["taille"]}  |  Région : {e["region"]}')
print(f'  Chiffre d\'affaires    : {e["ca"]/1e6:.2f} M DH')
print(f'  Délai client          : {e["delai_client_j"]:.0f} jours')
print(f'  Délai fournisseur     : {e["delai_fournisseur_j"]:.0f} jours')
print('─'*60)
print(f'  BFR comptable         : {bfr_comptable/1e6:.3f} M DH')
print(f'  BFR PRÉDIT (modèle)   : {bfr_predit/1e6:.3f} M DH  ⟵ prédiction ML')
print(f'  Écart comptable/ML    : {(bfr_predit - bfr_comptable)/1e6:+.3f} M DH')
print('─'*60)
print(f'  Ratio BFR/CA          : {ratio_bfr_ca:.1f}%')
print(f'  Diagnostic            : {diag_label}')
print('─'*60)
print(f'  🏦 Crédit de fonctionnement recommandé : {credit_recommande/1e6:.3f} M DH')
print('═'*60)

# ── Visualisation — Rapport entreprise individuelle ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Rapport BFR Prévisionnel — {e["nom"]}', fontsize=14, color='white', y=1.01)

# ① Décomposition du BFR
composants = {
    'Stocks (+)':          e['stocks']/1e6,
    'Créances clients (+)':e['creances_clients']/1e6,
    'Dettes four. (-)':   -e['dettes_fournisseurs']/1e6,
    'Dettes fisc. (-)':   -e['dettes_fiscales']/1e6,
}
colors_comp = ['#4ef7a0' if v > 0 else '#f74e8e' for v in composants.values()]
bars_comp = axes[0].bar(composants.keys(), composants.values(), color=colors_comp, alpha=0.85, edgecolor='none')
for bar, v in zip(bars_comp, composants.values()):
    axes[0].text(bar.get_x()+bar.get_width()/2,
                 v + 0.05 if v > 0 else v - 0.25,
                 f'{v:+.2f}M', ha='center', color='white', fontsize=9)
axes[0].axhline(0, color='white', lw=0.8)
axes[0].axhline(bfr_predit/1e6, color='#f7a04e', lw=2, linestyle='--',
                label=f'BFR prédit: {bfr_predit/1e6:.2f}M')
axes[0].set_title('Décomposition du BFR (M DH)')
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend(fontsize=9); axes[0].grid(axis='y', alpha=0.3)

# ② Jauge BFR/CA
theta = np.linspace(0, np.pi, 200)
r_ext, r_int = 1.0, 0.6
zones = [('Optimal\n0-8%', 8, '#4ef7a0'), ('Acceptable\n8-15%', 15, '#f7f74e'),
         ('Élevé\n15-22%', 22, '#f7a04e'), ('Critique\n22%+', 30, '#f74e8e')]
total = 30
start = 0
for label, end, col in zones:
    t1 = np.pi * (1 - start/total)
    t2 = np.pi * (1 - end/total)
    theta_z = np.linspace(t1, t2, 50)
    x = np.concatenate([r_int*np.cos(theta_z), r_ext*np.cos(theta_z[::-1])])
    y = np.concatenate([r_int*np.sin(theta_z), r_ext*np.sin(theta_z[::-1])])
    axes[1].fill(x, y, color=col, alpha=0.7)
    mid = (t1 + t2) / 2
    axes[1].text(0.82*np.cos(mid), 0.82*np.sin(mid), label.split('\n')[1],
                 ha='center', va='center', color='white', fontsize=7)
    start = end
# Aiguille
angle = np.pi * (1 - min(ratio_bfr_ca, 30)/total)
axes[1].annotate('', xy=(0.75*np.cos(angle), 0.75*np.sin(angle)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
axes[1].text(0, -0.15, f'{ratio_bfr_ca:.1f}%', ha='center', color='white',
             fontsize=14, fontweight='bold')
axes[1].set_xlim(-1.2, 1.2); axes[1].set_ylim(-0.4, 1.2)
axes[1].set_aspect('equal'); axes[1].axis('off')
axes[1].set_title('Jauge BFR/CA')

# ③ Plan de financement
labels_fin = ['FDR couvrant\nle BFR', 'Crédit fonct.\nrecommandé', 'BFR total\nprédit']
vals_fin   = [e['couverture_fdr']/1e6, credit_recommande/1e6, bfr_predit/1e6]
colors_fin = ['#4ef7a0', '#f7a04e', '#4e8ef7']
bars_fin   = axes[2].bar(labels_fin, vals_fin, color=colors_fin, alpha=0.85, edgecolor='none')
for bar, v in zip(bars_fin, vals_fin):
    axes[2].text(bar.get_x()+bar.get_width()/2, v+0.05, f'{v:.2f}M',
                 ha='center', color='white', fontsize=10, fontweight='bold')
axes[2].set_title('Plan de Financement du BFR (M DH)')
axes[2].set_ylabel('M DH')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/content/fig_rapport_entreprise.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

## 10. 📉 Analyse Scénarios & Optimisation du BFR

# ── Simulation de scénarios d'optimisation ─────────────────────────────────
scenarios = {
    'Situation actuelle':   {'delai_cl': e['delai_client_j'],    'delai_fo': e['delai_fournisseur_j'], 'stocks_ratio': e['couverture_stocks']},
    'Réduction délai client -15j': {'delai_cl': max(10, e['delai_client_j']-15),   'delai_fo': e['delai_fournisseur_j'], 'stocks_ratio': e['couverture_stocks']},
    'Allongement délai four. +10j': {'delai_cl': e['delai_client_j'],  'delai_fo': e['delai_fournisseur_j']+10, 'stocks_ratio': e['couverture_stocks']},
    'Optimisation stocks -20%': {'delai_cl': e['delai_client_j'],  'delai_fo': e['delai_fournisseur_j'], 'stocks_ratio': e['couverture_stocks']*0.80},
    'Scénario optimisé (tout)': {'delai_cl': max(10, e['delai_client_j']-15), 'delai_fo': e['delai_fournisseur_j']+10, 'stocks_ratio': e['couverture_stocks']*0.80},
}

bfr_scenarios = {}
for nom_sc, params_sc in scenarios.items():
    e_sc = e.copy()
    e_sc['stocks']            = e['ca'] * params_sc['stocks_ratio']
    e_sc['creances_clients']  = e['ca'] * (params_sc['delai_cl'] / 365)
    e_sc['dettes_fournisseurs']= e['ca'] * (params_sc['delai_fo'] / 365)
    e_sc['delai_client_j']    = params_sc['delai_cl']
    e_sc['delai_fournisseur_j']= params_sc['delai_fo']
    e_sc['couverture_stocks'] = params_sc['stocks_ratio']
    e_sc['intensite_client']  = e_sc['creances_clients'] / e_sc['ca']
    e_sc['levier_four']       = e_sc['dettes_fournisseurs'] / e_sc['ca']
    e_sc['bfr_exploitation']  = e_sc['stocks'] + e_sc['creances_clients'] - e_sc['dettes_fournisseurs']
    X_sc = pd.DataFrame([[e_sc[f] for f in FEATURES]], columns=FEATURES)
    bfr_sc = best_model.predict(X_sc)[0]
    bfr_scenarios[nom_sc] = bfr_sc

# Visualisation
fig, ax = plt.subplots(figsize=(13, 5))
colors_sc = ['#4e8ef7','#4ef7a0','#f7a04e','#a04ef7','#f74e8e']
names_sc  = list(bfr_scenarios.keys())
vals_sc   = [v/1e6 for v in bfr_scenarios.values()]
bars_sc   = ax.bar(names_sc, vals_sc, color=colors_sc, alpha=0.85, edgecolor='none')
for bar, v in zip(bars_sc, vals_sc):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.2f}M',
            ha='center', color='white', fontsize=9, fontweight='bold')
ax.axhline(vals_sc[0], color='white', lw=1.5, linestyle='--', alpha=0.7, label='Situation actuelle')
# Flèches d'économie
for i in range(1, len(vals_sc)):
    eco = vals_sc[0] - vals_sc[i]
    if eco > 0:
        ax.annotate(f'-{eco:.2f}M', xy=(i, vals_sc[i]+0.3), ha='center', color='#4ef7a0', fontsize=9)

ax.set_title("Analyse de Scénarios : Impact des Leviers d'Optimisation sur le BFR (M DH)", fontsize=12)
ax.set_ylabel('BFR prédit (M DH)')
ax.tick_params(axis='x', rotation=18)
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/content/fig_scenarios_optimisation.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()

print('\n📋 Résumé des scénarios :')
for nom_sc, v in bfr_scenarios.items():
    eco = bfr_predit - v
    print(f'  {nom_sc:42s} → BFR = {v/1e6:.3f} M DH  (économie: {eco/1e6:+.3f} M DH)')

## 11. 📋 Rapport Final & Export

# ── Export du dataset enrichi ──────────────────────────────────────────────
export_cols = ['secteur','taille','region','annee','ca','stocks','creances_clients',
               'dettes_fournisseurs','tresorerie','delai_client_j','delai_fournisseur_j',
               'ratio_endettement','marge_brute','bfr','bfr_predit','erreur_rel_%',
               'bfr_vs_ca_%','besoin_credit_fonct','diagnostic']
resultats[export_cols].to_csv('/content/predictions_bfr_maroc.csv', index=False, encoding='utf-8-sig')

# ── Rapport synthétique ────────────────────────────────────────────────────
print('═'*65)
print('  RAPPORT FINAL — PRÉDICTION BFR ENTREPRISES MAROCAINES')
print('  Sources : BAM (bilans sectoriels) + HCP (données macro)')
print('═'*65)
print(f'  Échantillon analysé     : {len(df)} entreprises simulées')
print(f'  Secteurs couverts       : {len(SECTEURS)}')
print(f'  Période                 : 2018–2023')
print(f'  Meilleur modèle         : {best_model_name}')
print(f'  R² (test)               : {results[best_model_name]["R²"]:.4f}')
print(f'  MAE                     : {results[best_model_name]["MAE (DH)"]/1e6:.3f} M DH')
print(f'  MAPE                    : {results[best_model_name]["MAPE (%)"]:.2f}%')
print('─'*65)
print('  TOP 3 variables prédictives :')
fi_top = pd.Series(best_model.feature_importances_, index=FEATURES).nlargest(3)
for feat, imp in fi_top.items():
    print(f'    ▸ {feat:30s} → {imp*100:.1f}%')
print('─'*65)
print(f'  BFR moyen prédit (portefeuille) : {resultats["bfr_predit"].mean()/1e6:.2f} M DH')
print(f'  Crédit fonct. moyen recommandé  : {resultats["besoin_credit_fonct"].mean()/1e6:.2f} M DH')
print(f'  % entreprises BFR critique      : {(resultats["diagnostic"]=="🔴 BFR Critique").mean()*100:.1f}%')
print('─'*65)
print('  Fichiers générés :')
print('    /content/predictions_bfr_maroc.csv')
print('    /content/fig_distribution_bfr.png')
print('    /content/fig_correlation.png')
print('    /content/fig_performances.png')
print('    /content/fig_reel_vs_predit.png')
print('    /content/fig_feature_importance.png')
print('    /content/fig_shap.png')
print('    /content/fig_dashboard_financier.png')
print('    /content/fig_rapport_entreprise.png')
print('    /content/fig_scenarios_optimisation.png')
print('═'*65)
