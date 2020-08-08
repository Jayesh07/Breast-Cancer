

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ClassificationReport



bcd = pd.read_csv("C:\\Users\\Jayesh\\Desktop\\data.csv")



print(bcd.head())
print(bcd.tail())
print(bcd.dtypes)



def b():

    sns.set(style="whitegrid", color_codes=True)

    sns.set(rc={'figure.figsize':(11.7,8.27)})

    sns.countplot('diagnosis',data=bcd,hue = 'diagnosis')

    sns.despine(offset=10, trim=True)
    
    plt.show()


b()



def v():
    
    sns.set(rc={'figure.figsize':(15,15)})
    sns.violinplot(x="diagnosis",y="radius_mean", hue="diagnosis", data=bcd)
    plt.show()
    
v()



data = bcd[['radius_mean']]
target = bcd['diagnosis']
train, test, train_labels, test_labels = train_test_split(data,
                                                          target,
                                                          test_size=0.15,
                                                          random_state=42)



print("\n")
print("1. By Naive Bayes algorithm")
print("\n")
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print("Accuracy of Naive Bayes classifier:"+str(accuracy_score(test_labels, preds)))

y_actu = bcd['diagnosis'].head(n=86) 
y_pred = preds
print("\n")
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)
print("\n")

visualizer = ClassificationReport(gnb, classes=['M','B'])
visualizer.fit(train, train_labels) 
visualizer.score(test,test_labels)
a = visualizer.poof() 

print("\n")

print("2. By Decision Tree algorithm")
print("\n")
clf = DecisionTreeClassifier()
model1=clf.fit(train,train_labels)
pred=clf.predict(test)

print("Accuracy of Decision Tree classifier:"+str(accuracy_score(test_labels,pred)))


y_actu = bcd['diagnosis'].head(n=86) 
y_pred = pred
print("\n")
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)
print("\n")

visualizer = ClassificationReport(clf, classes=['M','B'])
visualizer.fit(train, train_labels) 
visualizer.score(test,test_labels) 
c = visualizer.poof() 


print("\n")

print("3. By KNN algorithm")
print("\n")
knn = KNeighborsClassifier()
model2=knn.fit(train,train_labels)
p= knn.predict(test)

print("Accuracy of KNN classifier:"+str(accuracy_score(test_labels,p)))

y_actu = bcd['diagnosis'].head(n=86) 
y_pred = p
print("\n")
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)
print("\n")


visualizer = ClassificationReport(knn, classes=['M','B'])
visualizer.fit(train, train_labels) 
visualizer.score(test,test_labels) 
b = visualizer.poof() 




print("\n")

print("4. By Random Forest algorithm")
print("\n")
rfor = RandomForestClassifier()
model3=rfor.fit(train,train_labels)
ro=rfor.predict(test)
print("Accuracy of Random Forest classifier:"+str(accuracy_score(test_labels,ro)))



y_actu = bcd['diagnosis'].head(n=86) 
y_pred = ro
print("\n")
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)
print("\n")

visualizer = ClassificationReport(rfor, classes=['M','B'])
visualizer.fit(train, train_labels) 
visualizer.score(test,test_labels) 
c = visualizer.poof() 



