# Gaussian-Mixture-Model
In this project, we will be exploring a Credit Card Fraud dataset, and aim to develop a predictive model using Gaussian Mixture Model to determine a threshold to determine fraud samples. The process involves initial data pre-processing, Exploratory Data Analysis (EDA) includes visualizations like histograms for all the features and a class wise histogram. We define a Gaussian, GaussiaMixture and a custom_GMM class that trains on the honest data set. The project uses pandas, NumPy, and matplotlib for implementation and visualization, while it uses scipy.stats for the logsumexp function and also sklearn.cluster for importing the KMeans function, which is used only for initialisation of the parameters before implementing the GMM model which is coded from scratch. The project aims to identify a threshold for the log likelihood functions of the samples in the training data set.

## Procedure

### 1. Acquiring Data:
Obtain the dataset for the Credit Card fraud. Ensure the dataset includes all necessary features as required.
### 2. Data Pre-processing:
The data is scaled betweeen 0 and 1 across all columns.
### 3. Exploratory Data Analysis (EDA):
(a) Histogram 1: Visualize the high imbalance of the data by plotting the histogram class-wise.
(b) Histogram 2: We plot the feature wise histogram. We can observe visually about how each feature is or
almost is shaped like a bell-curve.
(c) df.describe( ) and df.info( ): We execute the two functions to gain more familiarity to the given data set.
### 4. Gaussian class : 
We define the Gaussian class where the pdf and logarithm of the pdf functions are stored.
### 5. GaussianMixture class : 
The GaussianMixture class is defined here, taking inspiration from the Gaussian-Mixture class in the sklearn package. The parameters are intiailized using KMeans from sklearn, and we then define the fit function by implementing the Expectation-Maximization Algorithm. We define the score_samples function to find the log likelihood of each sample in the trained data set.
### 6. custom_GMM: We initialize with attributes for storing GMMs, class priors, and log-likelihoods. The fit
function trains GMMs on input data based on target labels, calculating class priors and initializing GMMs. The predict_proba function gives the overall log likelihood. In the end, it exponentiates the log-joint likelihood and normalizes it to obtain the predicted probabilities. The shape of predicts is printed for debugging purposes, just in case.
### 7. Data splitting: 
The data is split into honest data and fraudulent data. Then we define training set by taking 80 percent of the honest data. We have to remove the columns we do not want to work with during training.
### 8. Plotting log-likelihoods of the trained data for threshold value deduction: 
After fitting the GMM, we take the log likelihoods of the trained data samples and print it against their density. We note where there is a significant drop in density, and consider that point as the threshold.

## References

The dataset can be found at Kaggle for free:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

To undestand the Gaussian Mixture Model, the following book was used
Bishop, C. M., and Nasrabadi, N. M. (2006). Pattern recognition and machine learning (Vol. 4) https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006

The following Jupyter notebooks were refered to:
1. https://www.kaggle.com/code/vineetkukreti/credit-card-fraud-detection/notebook
2. https://www.kaggle.com/code/bellashi/fraud-detection-using-gaussian-mixture-auc-97
