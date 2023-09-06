import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/ASUS/Desktop/projects/Mall_Customers.csv')

df.head()
df.columns
Index(['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)'],
      dtype='object')

df.rename(columns={'Annual Income (k$) ':'Income','Spending Score (1-100)':'Score'},inplace=True)

df.columns
Index(['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Score',
       'Clusters'],
      dtype='object')

df.head()


plt.scatter(df['Annual Income (k$)'],df.Score)
plt.show()


plt.scatter(df['Annual Income (k$)'],df.Score,marker='+',color='red')
plt.xlabel('ANNUAL INCOME')
plt.ylabel('SPENDING SCORE')
plt.show()


from sklearn.cluster import KMeans

km=KMeans(n_clusters=6)
results=km.fit_predict(df[['Annual Income (k$)','Score']])

results
array([4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
       4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2,
       4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 0, 5, 2, 5, 0, 5, 0, 5,
       2, 5, 0, 5, 0, 5, 0, 5, 0, 5, 2, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5,
       0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5,
       0, 5, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
       0, 3])

df['Clusters']=results

df.head()



df1=df[df['Clusters']==0]
df2=df[df['Clusters']==1]
df3=df[df['Clusters']==2]
df4=df[df['Clusters']==3]
df5=df[df['Clusters']==4]
df6=df[df['Clusters']==5]


plt.scatter(df1['Annual Income (k$)'],df1.Score,marker='*',color='red')
plt.scatter(df2['Annual Income (k$)'],df2.Score,marker='+',color='orange')
plt.scatter(df3['Annual Income (k$)'],df3.Score,marker='*',color='purple')
plt.scatter(df4['Annual Income (k$)'],df4.Score,marker='+',color='pink')
plt.scatter(df5['Annual Income (k$)'],df5.Score,marker='*',color='black')
plt.scatter(df6['Annual Income (k$)'],df6.Score,marker='+',color='blue')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
Text(0, 0.5, 'Spending Score')

#ELBOW METHOD

k_rng=range(1,10)
wcss=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Annual Income (k$)','Score']])
    wcss.append(km.inertia_)


wcss
[269981.28,
 181363.59595959593,
 106348.37306211118,
 73679.78903948836,
 44448.45544793371,
 37233.81451071001,
 30259.65720728547,
 25063.652515864094,
 21826.936303231654]

plt.plot(k_rng,wcss,marker='+')
plt.xlabel('k')
plt.ylabel('wcss')
plt.show()
