# =============================================================
# Obesity Risk Classification & Predictive Analytics
# Author: Omkar Pallerla | MS Business Analytics, ASU
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
COLORS = ['#4f9cf9', '#06d6a0', '#7c3aed', '#f59e0b', '#ef4444', '#ec4899']

# ── 1. LOAD DATA ────────────────────────────────────────────
df = pd.read_csv('ObesityDataSet.csv')
print(f"Shape: {df.shape}")
print(f"Target distribution:\n{df['NObeyesdad'].value_counts()}")

# ── 2. PREPROCESSING ─────────────────────────────────────────
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('NObeyesdad')

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df['target'] = le.fit_transform(df['NObeyesdad'])
target_names = le.classes_

X = df.drop(['NObeyesdad', 'target'], axis=1)
y = df['target']

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ── 3. SMOTE – handle class imbalance ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# ── 4. TRAIN MODELS ──────────────────────────────────────────
models = {
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=7)
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    cv  = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    results[name] = {'model': model, 'accuracy': acc, 'cv_score': cv, 'preds': y_pred}
    print(f"{name:25s} Acc={acc:.3f}  CV={cv:.3f}")

# ── 5. BEST MODEL EVALUATION ────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best = results[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best['preds'], target_names=target_names))

# ── 6. FEATURE IMPORTANCE ───────────────────────────────────
rf = results['Random Forest']['model']
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# ── 7. EXPORT RISK SCORES (BI Dashboard feed) ───────────────
proba = rf.predict_proba(X_test)
risk_df = X_test.copy()
risk_df['actual_class'] = y_test.values
risk_df['predicted_class'] = best['preds']
risk_df['max_probability'] = proba.max(axis=1)
for i, cls in enumerate(target_names):
    risk_df[f'prob_{cls}'] = proba[:, i]

# Map to 3-tier risk for dashboard
obese_classes = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
overweight_classes = ['Overweight_Level_I', 'Overweight_Level_II']
risk_df['risk_tier'] = risk_df['predicted_class'].apply(
    lambda x: 'HIGH' if target_names[x] in obese_classes
    else ('MEDIUM' if target_names[x] in overweight_classes else 'LOW')
)
risk_df.to_csv('outputs/risk_scores.csv', index=False)
print("Exported: outputs/risk_scores.csv")

# ── 8. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')

# Model comparison
ax = axes[0, 0]
names = list(results.keys())
accs  = [results[n]['accuracy'] for n in names]
bars  = ax.barh(names, accs, color=COLORS[:len(names)])
ax.set_xlim(0.7, 1.0)
ax.set_xlabel('Accuracy')
ax.set_title('Model Accuracy Comparison', color='white', pad=12)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{acc:.3f}', va='center', color='white', fontsize=10)

# Feature importance
ax = axes[0, 1]
top10 = importance_df.head(10)
ax.barh(top10['feature'][::-1], top10['importance'][::-1], color='#4f9cf9')
ax.set_title('Top 10 Feature Importances (Random Forest)', color='white', pad=12)

# Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names, yticklabels=target_names)
ax.set_title(f'Confusion Matrix — {best_name}', color='white', pad=12)
ax.tick_params(axis='x', rotation=45)

# Risk tier distribution
ax = axes[1, 1]
tier_counts = risk_df['risk_tier'].value_counts()
colors_tier = {'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#06d6a0'}
ax.pie(tier_counts, labels=tier_counts.index, colors=[colors_tier[t] for t in tier_counts.index],
       autopct='%1.1f%%', startangle=90)
ax.set_title('Patient Risk Tier Distribution', color='white', pad=12)

plt.tight_layout()
plt.savefig('outputs/obesity_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: outputs/obesity_analysis.png")
plt.show()
