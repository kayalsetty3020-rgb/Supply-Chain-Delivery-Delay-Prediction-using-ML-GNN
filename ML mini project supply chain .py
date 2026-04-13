#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================================================
# SUPPLY CHAIN DELIVERY DELAY PREDICTION
# IEEE LEVEL ML + GNN PROJECT
# =========================================================

# =========================================================
# 1 IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import networkx as nx
import shap

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 2 LOAD DATASET
# =========================================================

df = pd.read_csv("SCMS_Delivery_History_Dataset.csv")

print("Dataset Shape:", df.shape)

# =========================================================
# 3 DATA PREPROCESSING
# =========================================================

df = df.drop_duplicates()

date_cols = [
"PQ First Sent to Client Date",
"PO Sent to Vendor Date",
"Scheduled Delivery Date",
"Delivered to Client Date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# =========================================================
# 4 FEATURE ENGINEERING
# =========================================================

df["Delivery_Delay"] = (
df["Delivered to Client Date"] -
df["Scheduled Delivery Date"]
).dt.days

df["Late_Delivery"] = df["Delivery_Delay"].apply(lambda x: 1 if x > 0 else 0)

df["Vendor_Processing_Time"] = (
df["PO Sent to Vendor Date"] -
df["PQ First Sent to Client Date"]
).dt.days

df["Shipping_Time"] = (
df["Delivered to Client Date"] -
df["PO Sent to Vendor Date"]
).dt.days

# =========================================================
# 5 FEATURE SELECTION
# =========================================================

features = [
"Country",
"Shipment Mode",
"Vendor",
"Product Group",
"Line Item Quantity",
"Line Item Value",
"Pack Price",
"Freight Cost (USD)",
"Vendor_Processing_Time",
"Shipping_Time"
]

df_model = df[features + ["Late_Delivery"]]

df_model = df_model.dropna()

# =========================================================
# 6 ENCODE CATEGORICAL VARIABLES
# =========================================================

le = LabelEncoder()

for col in df_model.select_dtypes(include="object").columns:
    df_model[col] = le.fit_transform(df_model[col])

# =========================================================
# 7 EDA VISUALIZATION
# =========================================================

plt.figure()
sns.countplot(x="Late_Delivery", data=df_model)
plt.title("Late Delivery Distribution")
plt.show()

plt.figure()
sns.histplot(df_model["Freight Cost (USD)"], bins=30)
plt.title("Freight Cost Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df_model.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# =========================================================
# 8 SUPPLY CHAIN NETWORK GRAPH
# =========================================================

G = nx.Graph()

for i,row in df.iterrows():

    country = row["Country"]
    vendor = row["Vendor"]

    if pd.notna(country) and pd.notna(vendor):
        G.add_edge(country, vendor)

# Use small graph for visualization
G_small = G.subgraph(list(G.nodes)[:200])

plt.figure(figsize=(10,8))

pos = nx.spring_layout(G_small)

nx.draw_networkx(
    G_small,
    pos=pos,
    node_size=80,
    with_labels=False
)

plt.title("Supply Chain Network Graph")

plt.axis("off")

plt.show()

# =========================================================
# 9 TRAIN TEST SPLIT
# =========================================================

X = df_model.drop("Late_Delivery", axis=1)
y = df_model["Late_Delivery"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# =========================================================
# 10 HANDLE CLASS IMBALANCE
# =========================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# =========================================================
# 11 MACHINE LEARNING MODELS
# =========================================================

models = {

"Logistic Regression": LogisticRegression(max_iter=1000),

"Decision Tree": DecisionTreeClassifier(),

"Random Forest": RandomForestClassifier(),

"XGBoost": XGBClassifier(
use_label_encoder=False,
eval_metric="logloss"
)

}

results = {}

for name, model in models.items():

    model.fit(X_train_smote, y_train_smote)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    results[name] = acc

    print("\nModel:", name)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

# =========================================================
# 12 HYPERPARAMETER TUNING
# =========================================================

param_grid = {

"n_estimators":[100,200],

"max_depth":[5,10]

}

grid = GridSearchCV(
RandomForestClassifier(),
param_grid,
cv=3,
n_jobs=-1
)

grid.fit(X_train_smote, y_train_smote)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# =========================================================
# 13 MODEL EVALUATION
# =========================================================

preds = best_model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="ROC Curve (area=%0.2f)" % roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.legend()

plt.title("ROC Curve")

plt.show()

# =========================================================
# 14 FEATURE IMPORTANCE
# =========================================================

importance = pd.Series(best_model.feature_importances_, index=X.columns)

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")

plt.show()

# =========================================================
# 15 EXPLAINABLE AI (SHAP)
# =========================================================

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

# =========================================================
# 16 GRAPH NEURAL NETWORK
# =========================================================

node_features = torch.tensor(X.values, dtype=torch.float)

labels = torch.tensor(y.values, dtype=torch.long)

edge_index = torch.tensor([
[i for i in range(len(X)-1)],
[i+1 for i in range(len(X)-1)]
], dtype=torch.long)

data = Data(x=node_features, edge_index=edge_index, y=labels)

class GNN(torch.nn.Module):

    def __init__(self):

        super(GNN, self).__init__()

        self.conv1 = GCNConv(node_features.shape[1], 16)

        self.conv2 = GCNConv(16, 2)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):

    optimizer.zero_grad()

    out = model(data)

    loss = F.nll_loss(out, data.y)

    loss.backward()

    optimizer.step()

print("GNN Training Complete")

_, pred = model(data).max(dim=1)

correct = int(pred.eq(data.y).sum())

gnn_accuracy = correct / len(data.y)

print("GNN Accuracy:", gnn_accuracy)

# =========================================================
# 17 FINAL RESULTS
# =========================================================

print("\nMODEL COMPARISON")

for k,v in results.items():
    print(k, "Accuracy:", v)

print("\nBest Model:", best_model)

print("GNN Accuracy:", gnn_accuracy)

print("\nPROJECT COMPLETED SUCCESSFULLY")


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Load dataset
df = pd.read_csv("SCMS_Delivery_History_Dataset.csv")

# Convert date columns
date_cols = [
'PQ First Sent to Client Date',
'PO Sent to Vendor Date',
'Scheduled Delivery Date',
'Delivered to Client Date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Create delivery delay column
df['Delivery_Delay'] = (
    df['Delivered to Client Date'] -
    df['Scheduled Delivery Date']
).dt.days

# Create month column
df['Month'] = df['Scheduled Delivery Date'].dt.to_period('M').astype(str)

# Group by month and shipment mode
grouped = df.groupby(['Month', 'Shipment Mode'])['Delivery_Delay'].mean().reset_index()

# Plot
plt.figure(figsize=(14,6))

sns.lineplot(
    data=grouped,
    x='Month',
    y='Delivery_Delay',
    hue='Shipment Mode',
    marker='o'
)

plt.title('Average Delivery Delay Over Time by Shipment Mode')
plt.xlabel('Expected Delivery Month')
plt.ylabel('Average Delivery Delay (days)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




