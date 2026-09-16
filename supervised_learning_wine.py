#Partie_1  Problème de Régression (Prédiction Continue)

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


print("Preparation des donnees")
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = X['alcohol']          # cible
X = X.drop(columns=['alcohol'])  # variables explicatives

# Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)



# Régression Linéaire 
# Entrainement du modele


print("Entrainement du modele de regression lineaire")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)
print("Régression Linéaire")

mse_lin = mean_squared_error(y_test, y_pred_lin)
r2 = r2_score(y_test, y_pred_lin)
print("MSE :", mse_lin)
print("R² :", r2)


#Standardisation pour SVR
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)



# GridSearchCV pour trouver les meilleurs hyper-paramètres

print("Trouver les meilleure hyper-parametre")

param_grid = {
'C': [0.1, 1, 10, 100],
'gamma': [0.01, 0.1, 1,10]}

# Modèle SVR
svr = SVR(kernel='rbf')

# Recherche des meilleurs hyperparamètres
grid = GridSearchCV(
    svr,
    param_grid,
    cv=5,
    scoring='r2'
)


# Entraînement
grid.fit(X_train_s, y_train)

best_svr = grid.best_estimator_
y_pred_svr = best_svr.predict(X_test_s)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("SVR (RBF)")
print("Meilleurs paramètres :", grid.best_params_)
print("MSE :", mse_svr)
print("R² :", r2_svr)

print("comparaison entre le modele Regression Lineaire et Regression SVM (SVR)")
if mse_svr < mse_lin:
    print(" Le modele SVR offre la meilleure prediction du taux d alcool.")
else:
    print("La regression lineaire offre la meilleure prediction du taux d alcool.")



#partie2

#preparation des donnees

wine = load_wine()
X = wine.data
y = wine.target

# Filtrage des classes 0 et 1
classe_0_1 = (y == 0) | (y == 1)
X = X[classe_0_1]
y = y[classe_0_1]

# Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, random_state=42 )

# Standardisation
scaler2 = StandardScaler()
X_train2_s = scaler2.fit_transform(X_train2)
X_test2_s = scaler2.transform(X_test2)

#1. Régression Logistique : 
print('Application du modele de regression logistique')

#regression logistique

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train2_s, y_train2)

y_pred_log = log_reg.predict(X_test2_s)

print("Régression Logistique Accuracy et matrice de confusion")
print("Accuracy :", accuracy_score(y_test2, y_pred_log))
print("Matrice de confusion :\n", confusion_matrix(y_test2, y_pred_log))

#matrice de confusion et accuracy de la regression logistique

cm_log = confusion_matrix(y_test2, y_pred_log)
acc_log = accuracy_score(y_test2, y_pred_log)

# SVM Linéaire 

print("application du modele SVM Linéaire")

C_val = [0.1, 1, 100]

for C in [0.1, 1, 100]:
    svm_lin = SVC(kernel="linear", C=C)
    svm_lin.fit(X_train2_s, y_train2)
    y_pred_svm = svm_lin.predict(X_test2_s)
    acc_svm = accuracy_score(y_test2, y_pred_svm)
    print(f"C={C} | Accuracy :", acc_svm)
    cm_svm=confusion_matrix(y_test2, y_pred_svm)
    print(f"SVM Lineaire (C={C}) = Accuracy : {acc_svm},matrice_confu :{cm_svm}")



# Visualisation des frontières
print("visualisation  des frontières de décision de la Régression Logistique et du SVM ") 


X_visu = wine.data[:, [0, 12]][classe_0_1]
y_visu = wine.target[classe_0_1]

X_visu = StandardScaler().fit_transform(X_visu)

log_reg_visu = LogisticRegression(max_iter=1000)
svm_visu = SVC(kernel="linear", C=1)

log_reg_visu.fit(X_visu, y_visu)
svm_visu.fit(X_visu, y_visu)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plot_decision_regions(X_visu, y_visu, clf=log_reg_visu)
plt.title("Régression Logistique")

plt.subplot(1, 2, 2)
plot_decision_regions(X_visu, y_visu, clf=svm_visu)
plt.title("SVM Linéaire")

plt.show()



#partie 3

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

wine = load_wine()
X3 = wine.data
y3 = wine.target   

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X3, y3, test_size=0.3, random_state=42 )

scaler3 = StandardScaler()
X_train3_s = scaler3.fit_transform(X_train3)
X_test3_s = scaler3.transform(X_test3)



print("Entrainement de l'arbre de decision ")

cart = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
cart.fit(X_train3, y_train3)
y_pred_tree = cart.predict(X_test3)



print("arbre de decision Accuracy, Classification Report et Matrice de Confusion ")

print("Arbre de Décision Accuracy :", accuracy_score(y_test3, y_pred_tree))
print("Arbre de Décision Classification Report :",classification_report(y_test3, y_pred_tree))
print("Matrice de Confusion :\n", confusion_matrix(y_test3, y_pred_tree))


#affichage graphique
print("l'affichage graphique de l'arbre")

plt.figure(figsize=(20, 10))
plot_tree(cart, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True)
plt.title("Arbre de Décision CART")
plt.show()


#Foret aleatoire
#Entrainement 
print("Entrainement de la foret aleatoire")

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train3, y_train3)

y_pred_rf = rf.predict(X_test3)

print(" Foret Aleatoire Accuracy et Matrice de Confusion ")
print("Forêt Aléatoire Accuracy :", accuracy_score(y_test3, y_pred_rf))
print(classification_report(y_test3, y_pred_rf))
print("Matrice de Confusion :\n", confusion_matrix(y_test3, y_pred_rf))



print("Entrainement du modele deSVM Non-Linéaire avec noyeau RBF" )

svm_rbf = SVC(kernel="rbf", C=10, gamma=0.1)
svm_rbf.fit(X_train3_s, y_train3)
y_pred_rbf = svm_rbf.predict(X_test3_s)



print("SVM Non-Lineaire (RBF) Accuracy :", accuracy_score(y_test3, y_pred_rbf))

print("Application d'un reseau de neuron")

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

mlp.fit(X_train3_s, y_train3)

y_pred_mlp = mlp.predict(X_test3_s)



print("Réseau de Neurones")
print("Accuracy :", accuracy_score(y_test3, y_pred_mlp))
print("Reseau de Neurones (MLP) précision, rappel, F1-score")
print("Rapport de classification :",classification_report(y_test3, y_pred_mlp))