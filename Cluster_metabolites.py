import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import shap

def load_csv():
    metabolite_strains = pd.read_csv("final_metabolite_data.csv")
    print(metabolite_strains.columns)

    metabolite_strains = metabolite_strains.dropna()

    met_family=metabolite_strains["Category"].values

    X = np.concatenate([metabolite_strains["WT"].values.reshape(-1,1),metabolite_strains["Isoprene producer_642"].values.reshape(-1,1),metabolite_strains["Isoprene producer_704"].values.reshape(-1,1),metabolite_strains["Isoprene producer_731"].values.reshape(-1,1)],axis=1)
    print(X)
    y = met_family
    return X,y

def classify_and_visualize(X,y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ----- Train Logistic Regression -----
    #model = LogisticRegression()

    #model=DecisionTreeClassifier()
    model=RandomForestClassifier()
    #model=SVC(kernel='poly', degree=3)
    #model=LinearSVC()
    #model=HistGradientBoostingClassifier()
    model.fit(X_train, y_train)

    # ----- Predict -----
    y_pred = model.predict(X_test)

    # ----- Evaluate -----
    print("Accuracy:", accuracy_score(y_test, y_pred))

    #f1 = f1_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))


    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_test)

    print("Explained variance:", pca.explained_variance_ratio_)


    n_clusters = len(set(y))


    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_pred_encoded = le.fit_transform(y_pred)


    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)


    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred_encoded, cmap='tab10', s=30)
    plt.title("PCA — Colored by Predicted Labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    #plt.colorbar(label="Class")
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y_test_encoded, cmap='tab10', s=30)
    plt.title("PCA — Colored by True Labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    #plt.colorbar(label="Class")
    plt.show()

    return X_train,X_test,model

def shap_explain(model,X_train,X_test):

    feature_names = ['WT',
                     'Isoprene producer_642',
                     'Isoprene producer_704',
                     'Isoprene producer_731']

    explainer = shap.Explainer(model, X_train, feature_names=feature_names)
    shap_values = explainer(X_test)

    class_names = model.classes_
    print(class_names)

    # Plot for specific class
    plt.title(f"SHAP Beeswarm — Class: Organic acid")
    shap.plots.beeswarm(shap_values[:, :, list(class_names).index("Organic acid")])

    plt.title(f"SHAP Beeswarm — Class: Amino acid and nitrogenous compounds")
    shap.plots.beeswarm(shap_values[:, :, list(class_names).index("Amino acid and nitrogenous compounds")])

    plt.title(f"Carbohydrates and Phosphorylated Intermediates")
    shap.plots.beeswarm(shap_values[:, :, list(class_names).index("Carbohydrates and Phosphorylated Intermediates")])

if __name__ == "__main__":
    X,y=load_csv()
    X_train,X_test,model=classify_and_visualize(X,y)
    shap_explain(model,X_train,X_test)

