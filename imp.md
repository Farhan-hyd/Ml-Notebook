# Imp concepts

## Practicals

### 1. Data Preprocessing and Exploring

Data preprocessing is the process of transforming raw data into a useful, understandable format. Real-world or raw data usually has inconsistent formatting, human errors, and can also be incomplete. Data preprocessing resolves such issues and makes datasets more complete and efficient to perform data analysis.
Links:

- <https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/>
- <https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114#1c09>
- <https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/>

[code snippet](.\ml.ipynb#Data-Preprocessing) and important libraries:

- [pandas](https://regenerativetoday.com/30-very-useful-pandas-functions-for-everyday-data-analysis-tasks/)
- [numpy](https://www.codingninjas.com/codestudio/library/important-numpy-functions-for-ml)
- [matplotlib](https://medium.com/mlpoint/matplotlib-for-machine-learning-6b5fcd4fbbc7)
- [seaborn](https://seaborn.pydata.org/tutorial/introduction)

### 2,3. Regression And Classification

Regression is a statistical method for estimating the relationships among variables. It is mostly used for forecasting and finding out cause and effect relationship between variables. Regression techniques mostly differ based on the number of independent variables and the type of relationship between the independent and dependent variables.
Classification is a supervised machine learning technique which is used to classify the data into different classes. Classification is used to predict the class of given data points. Classification is a predictive analysis. It is used to find out the relationship between the independent and dependent variables.
Regression differ from classification in the sense that regression is used to predict the continuous value output while classification is used to predict the discrete value output.
Links:

- <https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86>
- <https://medium.com/data-science-group-iitr/logistic-regression-simplified-9b4efe801389>
- <https://www.geeksforgeeks.org/multivariate-regression/>

[code snippet](.\ml.ipynb#Regression) and important libraries:

- [sklearn: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [sklearn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [sklearn: mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [sklearn: mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

### 4. Cart

CART stands for Classification and Regression Trees. It is a decision tree algorithm that can be used for both classification and regression predictive problems. It is a non-parametric supervised learning method used for problems involving classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
Links:

- <https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052>

[code snippet](.\ml.ipynb#CART) and important libraries:

- [sklearn: DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [sklearn: plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)

### 5. SVM

SVM stands for Support Vector Machine. It is a supervised machine learning algorithm which can be used for both classification or regression challenges. However, it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well.
Links:

- <https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47>

[code snippet](.\ml.ipynb#SVM) and important libraries:

- [sklearn: SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [sklearn: confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### 6. Graph based clustering

Graph based clustering is a type of unsupervised machine learning algorithm that is used to find groups of similar objects based on their features. It is a type of clustering algorithm that is used to find groups of similar objects based on their features. It is a type of clustering algorithm that is used to find groups of similar objects based on their features.
Links:

- <https://www.geeksforgeeks.org/graph-clustering-methods-in-data-mining/>

[code snippet](.\ml.ipynb#Graph-based-clustering) and important libraries:
  
- [sklearn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [sklearn: SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

### 7. DBSCAN

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. It is a clustering algorithm that is used to find groups of similar objects based on their features. It is a type of clustering algorithm that is used to find groups of similar objects based on their features. It is a type of clustering algorithm that is used to find groups of similar objects based on their features.
Links:

- <https://towardsdatascience.com/explaining-dbscan-clustering-18eaf5c83b31>
- <https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan>

[code snippet](.\ml.ipynb#DBSCAN) and important libraries:
  
- [sklearn: DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

### 8. Baggging

Bagging stands for Bootstrap Aggregation. It is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting.
Links:

- <https://www.simplilearn.com/tutorials/machine-learning-tutorial/bagging-in-machine-learning>

[code snippet](.\ml.ipynb#Bagging) and important libraries:
  
- [sklearn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### 9. Boosting

Boosting is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones. Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy.
Links:

- <https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/>

[code snippet](.\ml.ipynb#Boosting) and important libraries:
  
- [sklearn: GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

### 10. PCA

Dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables.
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

PCA is implemented in [DBSCAN](.\ml.ipynb#DBSCAN) and important libraries:

- [sklearn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
