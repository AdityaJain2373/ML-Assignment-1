from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_linear_acc= pca.fit_transform(linear_samples)
plt.scatter(pca_linear_acc[:,0],pca_linear_acc[:,1],marker='.',c='r')
plt.grid(True)
plt.show()

