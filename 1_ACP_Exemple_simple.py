import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from affichage_acp import my_biplot

print('\n#############################')
print('# 1.1 Pré-étude des données #')
print('#############################\n')

# On charge les données
X = pd.read_csv("notes.csv", sep=';', index_col=0).transpose()

# On crée les vecteurs des noms des individus et des variables
nomi = list(X.index)  # noms des individus
nomv = list(X.columns)  # noms des variables

# On représente graphiquement la matière "français"
plt.figure(figsize=(10, 6))
plt.hist(X['fran'], edgecolor='cyan', bins=9)
plt.title('Distribution des notes en français')
plt.xlabel('Notes')
plt.ylabel('Nombre d\'élèves')
plt.show()

# On représente graphiquement la matière "latin"
plt.figure(figsize=(10, 6))
plt.hist(X['lati'], edgecolor='cyan', bins=9)
plt.title('Distribution des notes en latin')
plt.xlabel('Notes')
plt.ylabel('Nombre d\'élèves')
plt.show()

# On représente le nuage de points pour "mathématiques" et "sciences"
plt.figure(figsize=(10, 6))
plt.scatter(X['math'], X['scie'])
for i in range(len(nomi)):
    plt.text(X['math'].iloc[i], X['scie'].iloc[i], nomi[i])
plt.title('Nuage de points : Mathématiques vs Sciences')
plt.xlabel('Notes en mathématiques')
plt.ylabel('Notes en sciences')
plt.plot(range(4, 17), range(4, 17), color='cyan', linestyle='--', label='Indices = Colonnes')
plt.show()

# On représente le nuage de points pour "mathématiques" et "dessin"
plt.figure(figsize=(10, 6))
plt.scatter(X['math'], X['d-m '])
for i in range(len(nomi)):
    plt.text(X['math'].iloc[i], X['d-m '].iloc[i], nomi[i])
plt.title('Nuage de points : Mathématiques vs Dessin')
plt.xlabel('Notes en mathématiques')
plt.ylabel('Notes en dessin')
plt.show()

print('\n#############################')
print('#    1.2 Calcul de l’ACP    #')
print('#############################\n')

# On initialise l'objet PCA
p = len(nomv)  # Nombre de variables
acp = PCA(n_components=p)
cc = acp.fit_transform(X)

print('\n#############################')
print('#    1.3 Représentation     #')
print('#############################\n')

"""
Affichage de l’évolution de l’inertie expliquée (variance) cumulée selon le nombre d’axes (scree plot) : 
Pour choisir le nombre d’axes à conserver, on utilise un  plot qui montre la part de variance expliquée par chaque
composante principale.
"""

# Calcul de l'inertie expliquée cumulée
inertie_cumulee = np.cumsum(acp.explained_variance_ratio_)

# Scree plot
plt.figure(figsize=(10, 7))
plt.plot(range(1, p + 1), inertie_cumulee, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Inertie expliquée cumulée')
plt.grid(True)
plt.show()

"""
Représenter les individus dans les plans E1 ∪ E2 et E1 ∪ E3 (représentation 1) : Ces graphiques montrent la projection 
des individus sur les deux premiers axes principaux et sur le premier et le troisième axe principal.
"""

# Représentation des individus - Plan E1 ∪ E2
plt.scatter(cc[:, 0], cc[:, 1])
for i, nom in enumerate(nomi):
    plt.text(cc[i, 0], cc[i, 1], nom)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection des individus - Plan PC1 ∪ PC2')
plt.show()

# Représentation des individus - Plan E1 ∪ E3
plt.scatter(cc[:, 0], cc[:, 2])
for i, nom in enumerate(nomi):
    plt.text(cc[i, 0], cc[i, 2], nom)
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.title('Projection des individus - Plan PC1 ∪ PC3')
plt.show()

# Calcul des variances pour chaque variable
variances = X.var(axis=0)

# Calcul des corrélations
correlations = acp.components_ * np.sqrt(acp.explained_variance_[:, np.newaxis] / variances.values)

# Création d'un DataFrame pour une meilleure visualisation
correlation_matrix = pd.DataFrame(correlations, columns=X.columns, index=['PC{}'.format(i + 1) for i in range(p)])

# Affichage des corrélations
print(correlation_matrix)

# Pour visualiser les corrélations sous forme de heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap des corrélations entre variables et composantes principales')
plt.show()

"""
la représentation simultanée des individus et des variables (biplot) dans les plans E1∪E2 et E1 ∪ E3 
(représentation 2) : Un biplot permet de visualiser à la fois la projection des individus et la contribution des 
variables sur les axes principaux.
"""

# Préparation des données pour le biplot de PC1 et PC2
score = cc[:, 0:2]  # Les scores des individus sur les deux premiers axes principaux
coeff = np.transpose(acp.components_[0:2, :])  # Les coefficients des variables sur les deux premiers axes principaux

# Appel de la fonction my_biplot
my_biplot(score=score, coeff=coeff, coeff_labels=nomv, score_labels=nomi, nomx="PC1", nomy="PC2")

# Préparation des données pour le biplot de PC1 et PC3
score2 = cc[:, [0, 2]]  # Les scores des individus sur les deux premiers axes principaux
coeff2 = np.transpose(
    acp.components_[[0, 2], :])  # Les coefficients des variables sur le premier axe et le troisième axe.

my_biplot(score=score2, coeff=coeff2, coeff_labels=nomv, score_labels=nomi, nomx="PC1", nomy="PC3")
