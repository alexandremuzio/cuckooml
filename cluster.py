from sklearn import cluster

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples, silhouette_score

#Clustering algorithms

class KMeans(object):
	def __init__(self, vectorized_data, n_clusters):
		self.vectorized_data = vectorized_data
		self.n_clusters = n_clusters

		self.clusterer = cluster.KMeans(init='k-means++', 
									n_clusters=self.n_clusters,
									n_init=10,
									precompute_distances='auto',
									n_jobs=-1, #all cpu used
									)

	def run_clustering(self):
		self.clusterer.fit(self.vectorized_data)

	def labels(self):
		return self.clusterer.labels_

	def vizualize_results_in_2d(self, reduced_data, **kwargs):
		#Code from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

		# Step size of the mesh. Decrease to increase the quality of the VQ.
		h = .015	 # point in the mesh [x_min, m_max]x[y_min, y_max].

		# Plot the decision boundary. For that, we will assign a color to each
		x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
		y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		# Obtain labels for each point in mesh. Use last trained model.
		Z = self.clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.figure(1)
		plt.clf()
		plt.imshow(Z, interpolation='nearest',
				   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
				   cmap=plt.cm.Paired,
				   aspect='auto', origin='lower')

		plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
		# Plot the centroids as a white X
		centroids = self.clusterer.cluster_centers_
		plt.scatter(centroids[:, 0], centroids[:, 1],
					marker='x', s=169, linewidths=3,
					color='w', zorder=10)
		plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
				  'Centroids are marked with white cross')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.xticks(())
		plt.yticks(())

		if (kwargs['no_display'] == True):
			plt.savefig(kwargs['img_name'])

		else:
			plt.show()

	def plot_silhouette(self, reduced_data, **kwargs):
		#Code from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

		# Create a subplot with 1 row and 2 columns
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax1.set_xlim([-0.1, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(reduced_data) + (self.n_clusters + 1) * 10])

		# # Initialize the clusterer with n_clusters value and a random generator
		# # seed of 10 for reproducibility.
		# clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = self.clusterer.fit_predict(reduced_data)

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(reduced_data, cluster_labels)
		print("For n_clusters =", self.n_clusters,
			  "The average silhouette_score is :", silhouette_avg)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)

		y_lower = 10
		for i in range(self.n_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
				sample_silhouette_values[cluster_labels == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.spectral(float(i) / self.n_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper),
							  0, ith_cluster_silhouette_values,
							  facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhoutte score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

		# 2nd Plot showing the actual clusters formed
		colors = cm.spectral(cluster_labels.astype(float) / self.n_clusters)
		ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
					c=colors)

		# Labeling the clusters
		centers = self.clusterer.cluster_centers_
		# Draw white circles at cluster centers
		ax2.scatter(centers[:, 0], centers[:, 1],
					marker='o', c="white", alpha=1, s=200)

		for i, c in enumerate(centers):
			ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

		ax2.set_title("The visualization of the clustered data.")
		ax2.set_xlabel("Feature space for the 1st feature")
		ax2.set_ylabel("Feature space for the 2nd feature")

		plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
					  "with n_clusters = %d" % self.n_clusters),
					 fontsize=14, fontweight='bold')

		if kwargs['no_display'] == True:
			plt.savefig(kwargs['img_name'])

		else:
			plt.show()