"""Training the GMM model, taking number of components as 10"""

n_components = 10# We determine the hyperparameter by running it with different values many times
g = custom_GMM()
g.fit(np.array(X_train), np.array(y_train), n_components)

"""#### Plotting log-likelihoods of the trained data for threshold value deduction"""

log_likelihoods=g.predict_proba(np.array(X_train))

plt.hist(log_likelihoods, bins=10, density=True, alpha=0.7, color='blue')
plt.title('Distribution of Log-Likelihoods')
plt.xlabel('Log-Likelihood')
plt.ylabel('Density')
plt.xticks(np.linspace(min(log_likelihoods), max(log_likelihoods), 6))
plt.show()

"""Calcuating Statistical measures based on the log likelihood"""

mean_value = np.mean(log_likelihoods)
median_value = np.median(log_likelihoods)
std_dev = np.std(log_likelihoods)

print(f"Mean: {mean_value}, Median: {median_value}, Standard Deviation: {std_dev}")

"""Thus, we can guess the threshold around the point where the density dips drastically, somewhere between 0.000085 and 0.0001709. The anomalies can be stored by taking the samples from the trained data set with log likelihood values less than the threshold."""