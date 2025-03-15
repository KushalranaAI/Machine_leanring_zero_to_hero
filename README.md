# Machine_leanring_zero_to_hero
This repository will help to understand the core-concepts of machine learning and mathematical/programming implementation of ML.
---

## 1. Data Ingestion & Wrangling

**Goal**: Collect raw data from various sources and convert it into a workable format.

- **Working with CSV Files**  
  - Code and notebooks that demonstrate reading, writing, and parsing CSV files.
  - Techniques for merging, concatenating, and cleaning CSV data.
- **Working with JSON & SQL**  
  - Notebooks that showcase reading/writing JSON, connecting to SQL databases, and executing queries.
  - Handling nested JSON structures and converting them to tabular formats.
- **API to DataFrame**  
  - Tutorials on making requests to REST APIs or public data APIs.
  - Parsing JSON/XML responses into pandas DataFrames.
  - Authentication, pagination, rate-limiting considerations.
- **Web Scraping**  
  - Scripts demonstrating HTML parsing (e.g., BeautifulSoup, requests, Selenium).
  - Extracting tables, text, and links for data collection.

---

## 2. Exploratory Data Analysis (EDA)

**Goal**: Understand data distributions, relationships, and patterns to guide further modeling steps.

- **Descriptive Statistics**  
  - Summaries of numerical data (mean, median, mode), distribution shape (skew, kurtosis).
  - Visualizing single variables via histograms, boxplots, etc.
- **Univariate Analysis**  
  - Detailed look at each variable in isolation (outliers, data range, unique values).
- **Bivariate & Multivariate Analysis**  
  - Scatterplots, correlation heatmaps, pair-plots for understanding relationships.
  - Categorical vs. numerical analysis, group comparisons.
- **Automated Data Profiling**  
  - Tools such as `pandas_profiling` or `Sweetviz` that generate an overview report.
  - Quick checks for missing values, duplicates, and summary stats.

---

## 3. Data Cleaning & Imputation

**Goal**: Handle missing values, fix errors, and ensure data consistency prior to modeling.

- **Complete Case Analysis**  
  - Dropping rows/columns with missing data entirely (and discussing trade-offs).
- **Imputing Numerical Data**  
  - Mean/median imputation, arbitrary value imputation, regression-based imputation.
  - Discussion of potential biases introduced by each method.
- **Handling Missing Categorical Data**  
  - Replacing missing values with the most frequent category or a special “missing” label.
- **Missing Indicator**  
  - Adding binary flags to indicate whether a value was missing.
- **KNN Imputer**  
  - Distance-based methods for estimating missing values from similar rows.
- **Iterative Imputer**  
  - More advanced regression-based iterative techniques (MICE, etc.).

---

## 4. Feature Engineering & Transformation

**Goal**: Modify, encode, or transform data features to improve model performance or interpretability.

- **Standardization & Normalization**  
  - Converting data to zero-mean/unit-variance, min-max scaling, robust scaling.
- **Categorical Encoding**  
  - Ordinal encoding, one-hot/dummy encoding, target encoding.
- **Column Transformer & Pipelines**  
  - Automating and sequencing transformations with scikit-learn.
  - Saving and loading pipeline objects (`.pkl`) for reproducibility.
- **Function Transformer & Power Transformer**  
  - Applying custom transformations or non-linear transformations (Box-Cox, Yeo-Johnson).
- **Binning & Binarization**  
  - Converting continuous variables into discrete bins (e.g., age groups).
  - Thresholding for binary features.
- **Handling Mixed Variables**  
  - Splitting columns that contain multiple pieces of info (e.g., “City, State”).
  - Text or categorical expansions.
- **Handling Date & Time**  
  - Parsing date strings, extracting features like day, month, year, hour, or day of the week.
  - Creating time-based features (lag variables, rolling averages).
- **Outlier Handling**  
  - Identifying outliers using Z-score, IQR, percentile cutoffs, domain knowledge.
  - Discussing when to trim, winsorize, or transform outliers.
- **Feature Construction & Splitting**  
  - Creating interaction features, polynomial features, or domain-specific transformations.

---

## 5. Dimensionality Reduction

**Goal**: Reduce the feature space while retaining valuable variance.

- **Principal Component Analysis (PCA)**  
  - Mathematical derivation of PCA, step-by-step coding examples.
  - Explained variance, scree plots, selecting the number of components.
- **Other Techniques** (optional)  
  - Autoencoders, t-SNE, UMAP (if included for advanced dimensionality reduction).

---

## 6. Regression

**Goal**: Predict continuous numerical values using various techniques.

- **Simple Linear Regression**  
  - Single-predictor examples, slope/intercept derivations, model evaluation.
- **Multiple Linear Regression**  
  - Matrix form derivation, real-world datasets, assumptions (linearity, multicollinearity).
- **Polynomial Regression**  
  - Extending linear regression with polynomial features, bias-variance considerations.
- **Gradient Descent (GD)**  
  - Understanding cost functions (MSE), partial derivatives, convergence criteria.
- **Types of Gradient Descent**  
  - Batch, Stochastic, and Mini-batch GD. Trade-offs in speed, stability, and convergence.
- **Regression Metrics**  
  - MSE, RMSE, MAE, R², Adjusted R², interpretation for model selection.
- **Regularized Linear Models**  
  - **Ridge**: L2 regularization, controlling coefficient size.  
  - **Lasso**: L1 regularization, feature selection by shrinking coefficients to zero.  
  - **Elastic Net**: Combination of L1 and L2, balancing both effects.
- **Practical Tips**  
  - Checking residuals, dealing with heteroscedasticity, and ways to improve model generalization.
---

## 7. Classification

**Goal**: Predict discrete outcomes (binary or multi-class).

- **Logistic Regression**  
  - Sigmoid function, interpretation of coefficients, decision boundaries.
- **Classification Metrics**  
  - Accuracy, precision, recall, F1-score, confusion matrix, ROC & AUC.
- **Extended Logistic Regression**  
  - Softmax for multi-class classification, polynomial & interaction terms.
  - Perceptron trick and deeper theoretical aspects.
- **Practical Tips**  
  - Handling class imbalance (SMOTE, class weights, undersampling/oversampling).
  - Calibration of probabilities.


---

## 8. Ensemble Methods

**Goal**: Combine multiple models or approaches to improve performance and robustness.

- **Random Forest**  
  - Bagging vs. Random Forest, feature importance calculation, out-of-bag (OOB) estimates.
- **AdaBoost**  
  - Concept of boosting, updating sample weights, interpreting weak learners.
- **Gradient Boosting**  
  - Stage-wise additive modeling, learning rates, loss functions, code demos.
- **Stacking & Blending**  
  - Combining different model outputs at a meta-layer, stacking regressors/classifiers, blending techniques.
- **Practical Tips**  
  - Hyperparameter tuning (max depth, learning rate, n_estimators), trade-offs between bias and variance.

---

## 9. Clustering (Unsupervised)

**Goal**: Group data without pre-labeled classes.

- **K-Means**  
  - From-scratch implementation, scikit-learn usage, elbow method, silhouette scores.
- **Other Clustering Methods** (optional)  
  - Hierarchical clustering, DBSCAN, Gaussian Mixtures.
- **Use Cases**  
  - Customer segmentation, anomaly detection, topic modeling (if using text).

---

## 10. Reinforcement Learning (RL)

**Goal**: Train agents to make sequential decisions in an environment to maximize a reward.

- **Basics of RL**  
  - Markov Decision Processes (MDP), states, actions, rewards, policy vs. value-based methods.
- **Value-Based Methods**  
  - Q-Learning, SARSA, Temporal-Difference learning.
- **Policy-Based Methods**  
  - REINFORCE, Actor-Critic approaches.
- **Deep RL**  
  - Deep Q-Networks (DQN), advanced topics like Double DQN, Dueling DQN.
- **Practical Considerations**  
  - Exploration vs. exploitation, hyperparameter tuning, environment setup (OpenAI Gym).

---

## 11. Transfer Learning

**Goal**: Leverage knowledge from one domain or dataset (often a large, well-labeled dataset) to improve performance on a related but different task.

- **Pre-trained Models**  
  - Using existing models (e.g., ResNet, BERT) for feature extraction or fine-tuning.
- **Fine-Tuning**  
  - Techniques for partially or fully retraining layers on new data.
  - Freezing certain layers vs. unfreezing entire networks.
- **Domain Adaptation**  
  - Aligning feature distributions from source domain to target domain.
- **Practical Tips**  
  - Selecting optimal layers to fine-tune, monitoring training for overfitting, adjusting learning rates.


