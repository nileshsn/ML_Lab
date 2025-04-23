def get(section):
    snippets = {
      "data_handling": """

print(df.head())
print(df.tail())
print(df.sample(5))
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe(include='all'))
#print rows and columns
df.shape[0]
df.shape[1]
df["column_name"].unique()
# to print unique values
df["column_name"].nunique()
# to print number of unique values
df["column_name"].value_counts()
# to print value counts
df["column_name"].isnull().sum()
# to print number of missing values
df["column_name"].isna().sum()
# to print number of missing values
df["column_name"].isna().any()
# to print if any missing values

#fetch data from sklearn
from sklearn.datasets import fetch_20newsgroups
train_data=fetch_20newsgroups(subset='train')
test_data=fetch_20newsgroups(subset='test')

from sklearn.datasets import load_iris
data = load_iris()

# Change column names
df.columns = ['new_column_name', 'new_column_name2', 'new_column_name3']

# Drop columns
df.drop('column_name', axis=1, inplace=True)

# Drop rows
df.dropna( subset=['column_name'], axis=0 ,inplace=True)
# axis=0 for rows, axis=1 for columns

# data replace in column
df['column_name'].replace('old_value', 'new_value', inplace=True)

# Grouping data
df.groupby('column_name').
agg({'another_column_name': 'mean'})
# Data types
print(df.dtypes)

# Accessing a column
print(df['column_name'])

# Checking for duplicates
print(df.duplicated().sum())

# Saving dataframe
df.to_csv("saved_file.csv", index=False)
df.to_excel("saved_file.xlsx", index=False)

#manual linear regression
x_mean = x_train['TV'].mean()
y_mean = y_train.mean()
num = ((x_train['TV']-x_mean)*(y_train-y_mean)).sum()
den = ((x_train['TV']-x_mean)**2).sum()
slope = num/den
intercept = y_mean - slope*x_mean
print(f"Slope: {slope}, Intercept: {intercept}")

#create the model using the calculated coeffiecients and make predictions on the testing set
y_pred = slope*x_test['TV'] + intercept
y_pred.head()

#manual r2 score
ss_t = ((y_test-y_test.mean())**2).sum()
ss_r = ((y_test-y_pred)**2).sum()
r2_man = 1-(ss_r/ss_t)
n=len(y_test)
p=1
adj_r2 = 1-((1-r2_man)*(n-1)/(n-p-1))

print("R2 (manual): ", r2_man)
print("Adjusted R2 (manual): ", adj_r2)


# Step 11: Adjusted R2 (using 5 features only)
x_reduced = xm_s[:, :5]
x_reduced_scaled = s.fit_transform(x_reduced)
xm_train_r, xm_test_r, ym_train_r, ym_test_r = train_test_split(x_reduced_scaled, ym, test_size=0.3, random_state=42)

model_r = LinearRegression()
model_r.fit(xm_train_r, ym_train_r)
y_pred_r = model_r.predict(xm_test_r)

r2_r = r2_score(ym_test_r, y_pred_r)
n = xm_test_r.shape[0]
k = xm_test_r.shape[1]
adj_r2 = 1 - (1 - r2_r) * (n - 1) / (n - k - 1)
print('Adjusted R2:', adj_r2)

""", 

"data_cleaning": """

# Handling Missing Values
df.isnull().sum()         # Check missing values
df.dropna(inplace=True)   # Drop rows with missing values
df.fillna(0, inplace=True)   # Fill missing values with 0
df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean
df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)  # Fill with mode
df.ffill(inplace=True)    # Forward fill
df.bfill(inplace=True)    # Backward fill

# Handling Missing Values with Random Normal Distribution
mean, std = dt['Age'].mean(), dt['Age'].std()
dt['Age'] = dt['Age'].apply(lambda x: np.random.normal(mean, std) if pd.isnull(x) else x)

# Create new feature IsAlone from FamilySize
# If FamilySize == 1 then IsAlone = True else IsAlone = False
dd['IsAlone']=1
dd['IsAlone'].loc[dd['FamilySize']>1]=0

# Create a new attribute Has_cabin using the attribute Cabin.
# if Cabin == NaN then Has_cabin = 0 else Has_cabin = 1
dd['Has_cabin'] = dd['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Renaming Columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Converting Data Types
# from object to numeric
df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')
df['column_name'] = df['column_name'].astype(float)

# from numeric to object
df['column_name'] = df['column_name'].astype(object)
df['column_name'] = df['column_name'].astype(str)
df['column_name'] = df['column_name'].astype(bool)
df['column_name'] = df['column_name'].astype(int)
df['column_name'] = pd.to_datetime(df['column_name'])

# Outlier Detection and Removal
from scipy import stats
z = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z < 3).all(axis=1)]

# Handling categorical variables
df['category_column'] = df['category_column'].str.lower()  # Lowercase
df['category_column'] = df['category_column'].str.strip()  # Remove spaces

# Replace values
df['column_name'].replace({'old_value': 'new_value'}, inplace=True)

# Encoding binary categorical values
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Handling invalid entries
df = df[df['age'] >= 0]  # Remove negative ages

# Standardizing text
df['city'] = df['city'].str.title()   # 'new york' -> 'New York'

# Custom cleaning function
def clean_text(text):
    text = text.lower().strip()
    return text

df['column_name'] = df['column_name'].apply(clean_text)

# Removing unwanted columns
df.drop(['unnecessary_column1', 'unnecessary_column2'], axis=1, inplace=True)

# Resetting Index after cleaning
df.reset_index(drop=True, inplace=True)

""",

"data_visualization": """


# Essential Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Basic Plots
df['column_name'].hist(bins=30)
plt.title('Histogram of column_name')
plt.xlabel('column_name')
plt.ylabel('Frequency')
plt.show()

dd[['Age', 'Parch', 'Fare']].hist(figsize=(10, 5), bins=20)
plt.show()

plt.hist(dd[dd['Survived']==0]['Age'], bins=20, edgecolor='black', color='red', label='Did not Survived')
plt.hist(dd[dd['Survived']==1]['Age'], bins=20, edgecolor='black', color='blue', label='Survived')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sns.histplot(data=dd, x='Age', hue='Survived', bins=20, kde=True)
plt.title("Age vs Survival")
plt.show()

ds.hist(figsize=(10, 5), bins=20)
plt.tight_layout()
plt.show()

df.plot(kind='box', figsize=(8,6))
plt.title('Boxplot of all numeric features')
plt.show()

df['column_name'].plot(kind='line')
plt.title('Line Plot')
plt.show()

df['column_name'].value_counts().plot(kind='bar')
plt.title('Bar Plot of column_name')
plt.show()

# Scatter plot
plt.scatter(df['feature1'], df['feature2'])
plt.title('Scatter Plot')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()

# Seaborn pairplot
sns.pairplot(df)
plt.show()

# Seaborn heatmap (Correlation Matrix)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Seaborn countplot
sns.countplot(x='categorical_column', data=df)
plt.title('Count Plot of categorical_column')
plt.show()

# Seaborn boxplot
sns.boxplot(x='categorical_column', y='numeric_column', data=df)
plt.title('Boxplot Grouped by Categorical Column')
plt.show()

# Grouped Bar Plot
df.groupby('categorical_column')['numeric_column'].mean().plot(kind='bar')
plt.title('Mean of numeric_column by categorical_column')
plt.show()

# Pie Chart
df['categorical_column'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart of categorical_column')
plt.show()

# Distribution Plot (distplot is deprecated, using displot or histplot)
sns.histplot(df['column_name'], kde=True)
plt.title('Distribution Plot')
plt.show()

# Tree Plot
plt.figure(figsize=(15, 10))
plot_tree(md, filled=True, feature_names=xd.columns, class_names=['0', '1'], max_depth=3)
plt.show()

# Image Plot (for image data)
for i in range(5):
    plt.imshow(x.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y.iloc[i]}")
    plt.show()

# Multiple Subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
sns.histplot(df['feature1'], ax=axs[0, 0])
sns.histplot(df['feature2'], ax=axs[0, 1])
sns.boxplot(y=df['feature3'], ax=axs[1, 0])
sns.scatterplot(x=df['feature1'], y=df['feature2'], ax=axs[1, 1])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x='Pclass', hue='Survived', data=dd, ax=ax[0])
sns.countplot(x='Sex', hue='Survived', data=dd, ax=ax[1])
plt.show()

# Scatter plot
for col in xm.columns:
    plt.scatter(xm[col], ym)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.show()

# Bar plot
cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
for cols in cols:
    sns.countplot(x=dd[cols])
    plt.show()

# Plot barchart of L-CORE, L-SURF,L-O2 and L-BP using 4X4 subplots - 3 Marks
#num_bins = 10
plt.figure(figsize=(10, 10))
col = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP']

for i, cols in enumerate(col, 1):
    plt.subplot(2, 2, i)
    if df[cols].dtype == 'object': 
        data = df[cols].value_counts()
        plt.bar(data.index, data.values)
        plt.title(cols)
        plt.xticks(rotation=45)
    else: 
        data = df[cols].dropna() 
        plt.hist(data, bins=10)
        plt.title(cols)

plt.tight_layout()
plt.show() 

""",

"data_preprocessing": """


# Essential Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# CountVectorizer Example
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
xcv_train = cv.fit_transform(train_sub.data)
xcv_train

# TF-IDF Vectorizer Example
tfidf = TfidfVectorizer(stop_words='english')
x_train_tf = tfidf.fit_transform(train_sub.data)
x_test_tf = tfidf.transform(test_sub.data)

mnb = MultinomialNB()
mnb.fit(x_train_tf, train_sub.target)
y_pred_tf = mnb.predict(x_test_tf)

# Splitting features and target
X = df.drop('target_column', axis=1)
y = df['target_column']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling - StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scaling - MinMaxScaler
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

# Encoding Categorical Features - LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# concat df_categorical with original df
xd=pd.concat([xd, pd.get_dummies(xd['Pclass'], prefix='Pclass', drop_first=True)], axis=1)
xd.drop('Pclass', axis=1, inplace=True)

# Imputation of Missing Values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# ColumnTransformer Example
numeric_features = ['num_col1', 'num_col2']
categorical_features = ['cat_col1', 'cat_col2']

# Feature Engineering Example
df['new_feature'] = df['feature1'] * df['feature2']

""",

"modeling": """


# Essential Imports
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Example: Initialize model and train
# Logistic Regression (Classification)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Linear Regression (Regression)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Decision Tree (Classification)
# we pass parameters to the model explain and give it in comments and code syntax also
# Decision Tree (Classification) - A tree-based model that splits the data into branches to make predictions based on feature values

dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42, min_samples_leaf=5)
dt_clf.fit(X_train, y_train)

# Create the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Support Vector Machine (Classification)
svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)

from sklearn.svm import SVC
svm = SVC(C=1)
svm.fit(xs_train, ys_train)

# K-Nearest Neighbors (Classification)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

# Naive Bayes (Classification)
nb = GaussianNB()
nb.fit(X_train, y_train)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Saving Trained Model
import joblib
joblib.dump(rf_clf, 'random_forest_model.pkl')

""",

"model_evaluation": """

# Essential Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Classification Metrics

# Predict on test set
y_pred = log_reg.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Precision
prec = precision_score(y_test, y_pred, average='binary')  # use 'macro' or 'weighted' for multiclass
print("Precision:", prec)

# Recall
rec = recall_score(y_test, y_pred, average='binary')
print("Recall:", rec)

# F1 Score
f1 = f1_score(y_test, y_pred, average='binary')
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))


# 2. Regression Metrics

# Predict
y_pred_reg = lin_reg.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred_reg)
print("Mean Squared Error:", mse)

# Root Mean Squared Error
rmse = mean_squared_error(y_test, y_pred_reg, squared=False)
print("Root Mean Squared Error:", rmse)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred_reg)
print("Mean Absolute Error:", mae)

# R2 Score
r2 = r2_score(y_test, y_pred_reg)
print("R2 Score:", r2)


# 3. Clustering Metrics

# Inertia (for KMeans)
print("KMeans Inertia:", kmeans.inertia_)

# Silhouette Score
sil_score = silhouette_score(X, kmeans.labels_)
print("Silhouette Score:", sil_score)


# 4. Cross Validation

from sklearn.model_selection import cross_val_score

# Cross Validation Accuracy
cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Cross Validation RMSE (for regression)
cv_rmse = cross_val_score(lin_reg, X, y, scoring='neg_root_mean_squared_error', cv=5)
print("Cross-validation RMSE:", -cv_rmse.mean())
""",

"kmeans": """

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Load iris dataset
iris = load_iris()

dir(iris)

df = pd.DataFrame(iris.data)

# Rename columns
df.rename(columns={
    0: "sepalLength",
    1: "sepalWidth", 
    2: "petalLength",
    3: "petalWidth"
}, inplace=True)

# Create pairplot
sns.pairplot(df[['sepalLength','sepalWidth','petalLength','petalWidth']])
plt.savefig('pairplot.png')
plt.close()

# Find optimal number of clusters using elbow method
k_range = range(1,10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10)
    km.fit_predict(df[['petalLength','petalWidth','sepalLength','sepalWidth']])
    sse.append(km.inertia_)

# Plot elbow curve
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method')
plt.savefig('elbow_curve.png')
plt.close()

# Fit KMeans with optimal clusters
km = KMeans(n_clusters=3, n_init=10)
clusters = km.fit_predict(df[['petalLength','petalWidth','sepalLength','sepalWidth']])

# Add cluster labels
df['target'] = clusters

# Visualize clusters with pairplot
sns.pairplot(df, hue='target', vars=['sepalLength','sepalWidth','petalLength','petalWidth'], palette='pastel')
plt.savefig('clusters_pairplot.png')
plt.close()

# Create scatter plot for clusters
df1 = df[df['target']==0]
df2 = df[df['target']==1] 
df3 = df[df['target']==2]

plt.figure(figsize=(10,6))
plt.scatter(df1['petalLength'], df1['petalWidth'], label="class0")
plt.scatter(df2['petalLength'], df2['petalWidth'], label="class1")
plt.scatter(df3['petalLength'], df3['petalWidth'], label="class2")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Clusters before scaling')
plt.legend()
plt.savefig('clusters_scatter.png')
plt.close()

# Scale features
scaler = MinMaxScaler()
features = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
for feature in features:
    df[feature] = scaler.fit_transform(df[[feature]])

# Visualize centroids after scaling
plt.figure(figsize=(10,6))
plt.scatter(df1['petalLength'], df1['petalWidth'], label="class0")
plt.scatter(df2['petalLength'], df2['petalWidth'], label="class1")
plt.scatter(df3['petalLength'], df3['petalWidth'], label="class2")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], 
            color="purple", marker="+", s=200, label="centroid")
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('Clusters with Centroids')
plt.legend()
plt.savefig('clusters_with_centroids.png')
plt.close()

# Calculate and print silhouette score
silhouette_avg = silhouette_score(
    df[['sepalLength','sepalWidth','petalLength','petalWidth']], 
    km.labels_
)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Save results to csv
df.to_csv('kmeans_results.csv', index=False)
"""
}
    return snippets.get(section, "Section not found. Available keys:\n" + ", ".join(snippets.keys()))