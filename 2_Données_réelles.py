from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# On charge les données
imgtmp = loadmat("Indian_pines_corrected.mat")
img = np.float32(imgtmp['indian_pines_corrected'])
maptmp = loadmat("Indian_pines_gt.mat")
map = maptmp['indian_pines_gt']

# On remodele l'image en 2D pour l'ACP
img_reshaped = img.reshape(-1, img.shape[-1])

# Afficher la carte
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(map)
plt.title('Carte de la zone Indian Pines')
plt.xlabel('Coordonnée Est-Ouest')
plt.ylabel('Coordonnée Nord-Sud')

# Afficher l'image
plt.subplot(1, 2, 2)
plt.imshow(img[:, :, 0])
plt.title('Première bande de l\'image corrigée de la zone Indian Pines')
plt.xlabel('Coordonnée Est-Ouest')
plt.ylabel('Coordonnée Nord-Sud')

plt.tight_layout()
plt.show()

# Appliquer l'ACP
pca = PCA()
img_pca = pca.fit_transform(img_reshaped)

# On évalue le nombre d'axes nécessaires
explained_variance = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)
plt.plot(cum_explained_variance)
plt.xlabel('Nombre d\'axes principaux')
plt.ylabel('Variance cumulative expliquée')
plt.title('Proportion de variance expliquée par l\'ACP')
plt.grid(True)
plt.show()

# On sélectionne le premier composant principal
premier_composant_principal = img_pca[:, 0]

# On remodele pour retrouver la forme d'image originale
composant_remodele = premier_composant_principal.reshape((145, 145))

fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# On visualise en tant qu'image en niveaux de gris
axs[0].imshow(composant_remodele, cmap='gray')
axs[0].set_title('Premier Axe Principal - Image Originale \nRemodelée en niveaux de gris')

# Affichage de l'image en col
img_col = img_pca[:, :3]
img_reshaped_col = img_col.reshape(145, 145, 3)
axs[1].imshow(img_reshaped_col)
axs[1].set_title("Troisième Axe Principal - Image Originale \nRemodelée en couleur sans normalisation")

# Affichage de l'image en col normalisé
axs[2].imshow((img_reshaped_col - np.min(img_reshaped_col)) / (np.max(img_reshaped_col) - np.min(img_reshaped_col)))
axs[2].set_title('Troisième Axe Principal - Image Originale \nRemodelée en couleur normalisation')

plt.show()

# Le nombre d'axes basé sur notre analyse
num_axes = 6
axes_selectionnes = img_pca[:, :num_axes]

# On détermine le nombre de lignes nécessaires pour afficher 4 graphiques par ligne
num_graphiques_par_ligne = 3
num_lignes = int(np.ceil(num_axes / num_graphiques_par_ligne))

# On visualise la projection sur les premiers axes principaux en couleur
fig, axes = plt.subplots(num_lignes, num_graphiques_par_ligne, figsize=(10, 8))
fig.subplots_adjust(hspace=0.5)

for i in range(num_axes):
    ligne = i // num_graphiques_par_ligne
    colonne = i % num_graphiques_par_ligne
    projection = axes_selectionnes[:, i].reshape((145, 145))
    # Normalisation des données entre 0 et 1
    projection = (projection - np.min(projection)) / (np.max(projection) - np.min(projection))
    axes[ligne, colonne].imshow(projection, cmap='Accent')
    axes[ligne, colonne].set_title(f'Axe principal {i + 1}')

# On supprime les graphiques inutilisés
for i in range(num_axes, num_lignes * num_graphiques_par_ligne):
    fig.delaxes(axes.flatten()[i])

plt.show()

plt.tight_layout()

plt.show()

# On compare avec la vérité terrain
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(map, cmap='jet')
ax.set_title('Comparaison avec la vérité du terrain')
cbar = plt.colorbar(im, ax=ax)

# On ajoute des annotations pour chaque classe
for i, nom_classe in enumerate([
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
]):
    cbar.ax.text(2.5, i / 16.0, nom_classe, fontsize=8, ha='left', va='center',
                 transform=cbar.ax.transAxes)

plt.show()

# Création d'une liste de couleurs pour les différentes signatures spectrales
colors = ['b', 'g', 'r', 'c', 'm']

# Création d'une figure avec 2 sous-graphiques
fig, axs = plt.subplots(2, figsize=(10, 12))

# On trace la signature spectrale du premier axe principal dans le premier sous-graphique
axs[0].plot(pca.components_[0, :], color='b')
axs[0].set_xlabel('Bande spectrale')
axs[0].set_ylabel('Valeur du coefficient')
axs[0].set_title('Signature spectrale du premier axe principal')

# Boucle sur les 5 premiers axes principaux pour le deuxième sous-graphique
for i in range(5):
    axs[1].plot(pca.components_[i, :], color=colors[i], label=f'Axe principal {i + 1}')

axs[1].set_xlabel('Bande spectrale')
axs[1].set_ylabel('Valeur du coefficient')
axs[1].set_title('Signature spectrale des 5 premiers axes principaux')
axs[1].legend()

# Affichage de la figure
plt.tight_layout()
plt.show()

#Bonus
# Choix du nombre de clusters
nombre_clusters = 5

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=nombre_clusters, n_init=10)
labels = kmeans.fit_predict(img_pca)

# Remodelage des labels pour les rendre compatibles avec la forme de l'image
labels_img = labels.reshape(img.shape[0], img.shape[1])

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.imshow(labels_img, cmap='viridis')
plt.title(f'Résultats de la segmentation avec K-means ({nombre_clusters} clusters)')
plt.colorbar(label='Numéro du cluster')
plt.show()
