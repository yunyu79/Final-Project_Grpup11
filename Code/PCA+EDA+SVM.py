#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns 

#%%
sounds = pd.read_csv("sounds.csv")
plt.matshow(sounds.corr())

#%%
X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]
#%%
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps

#%%
pca = PCA(n_components = 10)
pca.fit(X_scaled)
X_pca = pca.fit_transform(X_scaled)

plt.bar(range(1,len(pca.explained_variance_ )+1),pca.explained_variance_ )
plt.ylabel('Explained variance')
plt.xlabel('Components')
plt.plot(range(1,len(pca.explained_variance_ )+1),
         np.cumsum(pca.explained_variance_),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='upper left')

#%%
df_new = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
df_new['label'] = y
df_new.head()

#%%
X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size = 0.2, random_state = 0)

#%%
X_train = pd.DataFrame(X_train)
ax = sns.heatmap(X_pca.components_,
                 cmap='YlGnBu',
                 yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
                 xticklabels=list(X_train.columns),
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")

#%%
# First two of components 
plt.scatter(X_pca[:, 0], X_pca[:, 1])
# %%
# The first 10 components' explained_variance_ratio
pca.explained_variance_ratio_ *100