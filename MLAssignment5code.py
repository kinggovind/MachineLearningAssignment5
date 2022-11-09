import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

# 1. Principal Component Analysis

cc_dataset = pd.read_csv('CC.csv')
cc_dataset.head()
X = cc_dataset.iloc[:, 1:]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X = imputer.transform(X)
X = pd.DataFrame(X)
pca = PCA(2)
x_pca = pca.fit_transform(X)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, X.iloc[:, -1]], axis=1)
finaldf.head()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(finaldf)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(finaldf)
y_cluster_kmeans = km.predict(finaldf)

score = metrics.silhouette_score(finaldf, y_cluster_kmeans)
print(score)
X = cc_dataset.iloc[:, 1:]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)

X = imputer.transform(X)

print(X)
X = pd.DataFrame(X)
scaler = StandardScaler()
scaler.fit(X)

x_scaler = scaler.transform(X)

pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, cc_dataset[['TENURE']]], axis=1)
print(finaldf)

nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(finaldf)
y_cluster_kmeans = km.predict(finaldf)

score = metrics.silhouette_score(finaldf, y_cluster_kmeans)
print(score)


# 2. Use pd_speech_features.csv

speech_df = pd.read_csv('pd_speech_features.csv')
speech_df.head()
x = speech_df.iloc[:, 1:]
scaler = StandardScaler()
scaler.fit(x)
speech_x_scaler = scaler.transform(x)

pca = PCA(3)
speech_x_pca = pca.fit_transform(speech_x_scaler)
speech_df2 = pd.DataFrame(data=speech_x_pca)
speech_finaldf = pd.concat([speech_df2, speech_df[['class']]], axis=1)
print(speech_finaldf)

clf = SVC(kernel='linear')

x = speech_finaldf.iloc[:, :-1]
y = speech_finaldf.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)

print("svm accuracy =", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
# 3. Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k=2


iris_df = pd.read_csv("Iris.csv")

iris_df.head()
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(iris_df.iloc[:, :-1].values)
class_le = LabelEncoder()
y = class_le.fit_transform(iris_df['Species'].values)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y)
data = pd.DataFrame(X_train_lda)
data['class'] = y
data.columns = ["LD1", "LD2", "class"]
data.head()
markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show()