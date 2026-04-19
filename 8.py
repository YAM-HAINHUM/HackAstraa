import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target_names[iris.target]

print(data.head())
print(data.shape)
print("Unique species:", iris.target_names)

# Bar Plot
sns.countplot(x='Species', data=data)
plt.title('Bar Plot for 3 Species')
plt.show()

# Pie Chart
data['Species'].value_counts().plot.pie(
    explode=[0.05, 0.05, 0.05],
    autopct="%1.1f%%",
    shadow=True,
    figsize=(6, 6)
)
plt.title('Pie Chart of Species')
plt.ylabel("")
plt.show()

# Box Plot
sns.boxplot(x=data['petal length (cm)'])
plt.title('Boxplot of Petal Length')
plt.show()

sns.boxplot(x='Species', y='petal length (cm)', data=data)
plt.title('Petal Length for 3 Species')
plt.show()

# Histogram + PDF
sns.FacetGrid(data, hue="Species", height=5) \
    .map(sns.histplot, "petal length (cm)", kde=True) \
    .add_legend()
plt.title('Histogram and PDF of Petal Length')
plt.show()

# PDF and CDF for Setosa
setosa = data[data['Species'] == 'setosa']
counts, bin_edges = np.histogram(setosa['petal length (cm)'], bins=10, density=True)

pdf = counts / np.sum(counts)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label='PDF')
plt.plot(bin_edges[1:], cdf, label='CDF')
plt.xlabel('Petal Length')
plt.ylabel('Probability')
plt.title('PDF and CDF for Setosa Petal Length')
plt.legend()
plt.show()

# Scatter Plots
data.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)')
plt.title('Scatter Plot')
plt.show()

sns.set_style("whitegrid")
sns.FacetGrid(data, hue="Species", height=4) \
    .map(plt.scatter, 'sepal length (cm)', 'sepal width (cm)') \
    .add_legend()
plt.title('Scatter Plot by Species')
plt.show()

# Heatmap
sns.heatmap(data.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Line Chart
for species in data['Species'].unique():
    subset = data[data['Species'] == species]
    plt.plot(subset['petal length (cm)'], subset['petal width (cm)'], label=species)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Line Chart: Petal Length vs Petal Width by Species')
plt.legend()
plt.show()

# Word Cloud
text_data = " ".join(species for species in data['Species'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=100
).generate(text_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Iris Species Distribution')
plt.show()
