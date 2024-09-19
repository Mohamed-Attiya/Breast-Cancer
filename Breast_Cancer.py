# Import Python Libraries :
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data = pd.read_csv("E:\\Breast Cancer\\data.csv")

# Explore The Data :
print(data.head(5))
print("=====================================")
print(data.tail(5))
print("=====================================")
print(data.sample()) # One Random Data

# Data Description (Columns & Rows) :
print("=====================================")
print(data.shape)
print("=====================================")
print(data.info())
print("=====================================")
print(data.describe())
print("=====================================")
print(data.isnull().sum()) # = print(data.isna().sum())
print("=====================================")
print(data['diagnosis'].value_counts())
print("=====================================")
sns.countplot(data = data , x ='diagnosis')
plt.show()
print(data.columns)
print("=====================================")
data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)
print(data.columns)
print("=====================================")

# Convert diagnosis back to a categorical type for plotting
data['diagnosis'].replace({"M":"1","B":"0"},inplace = True)
print(data.head())
sns.countplot(data = data , x ='diagnosis')
plt.show()
print("=====================================")
print(data['diagnosis'].value_counts())
print("=====================================")
print(data.describe())
print("=====================================")
print(data.describe().T)
sns.pairplot(data,hue='diagnosis')
plt.figure(figsize = (20,20))
sns.heatmap(data.corr() , annot =  True , cmap = 'mako')

# Machine Learning (Spilitting The Data) :
X = data.drop(['diagnosis'],axis = 1)
y = data['diagnosis']

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.30 , random_state = 101) 

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train , y_train)
pred = log_model.predict(X_test)

from sklearn.metrics import classification_report , accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


# Confusion Matrix Without Normalization :
print('Confusion Matrix')

# Generate predictions
y_pred = log_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_model.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()   

# Confusion Matrix With Normalization :
print('Normalized Confusion Matrix')

# Generate predictions
y_pred = log_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_model.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

print("=====================================")
print(confusion_matrix(y_test , pred))
print("=====================================")
print(classification_report(y_test , pred))
print("=====================================")
print(accuracy_score(y_test,pred))
print("=====================================")


from sklearn.neighbors import KNeighborsClassifier
# Creating KNN Classifier Model:
knn = KNeighborsClassifier(n_neighbors=2)

# Fitting the Training Data:
knn.fit(X_train, y_train)

# Predicting on the Test Data:
pred = knn.predict(X_test)

# Confusion Matrix Without Normalization:
print('Confusion Matrix')
print("=====================================")

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred)

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

# Confusion Matrix With Normalization:
print('Normalized Confusion Matrix')

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred,normalize='true')

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

print("=====================================")
print(confusion_matrix(y_test , pred))
print("=====================================")
print(classification_report(y_test , pred))
print("=====================================")
print(accuracy_score(y_test,pred))
print("=====================================")


from sklearn.ensemble import RandomForestClassifier
# Making an object for RFC :
RF = RandomForestClassifier(n_estimators = 100 , random_state = 0)

# Fitting The Model : 
RF.fit(X_train , y_train)

# Prediciting on test set :
pred = RF.predict(X_test)

# Confusion Matrix Without Normalization:
print('Confusion Matrix')
print("=====================================")

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred)

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

# Confusion Matrix With Normalization:
print('Normalized Confusion Matrix')

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred,normalize='true')

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

print("=====================================")
print(confusion_matrix(y_test , pred))
print("=====================================")
print(classification_report(y_test , pred))
print("=====================================")
print(accuracy_score(y_test,pred))
print("=====================================")


from sklearn.tree import DecisionTreeClassifier
# Making an object of the classifier :
DT = DecisionTreeClassifier(random_state = 42).fit(X_train , y_train)

# Predicting on the test set :
pred = DT.predict(X_test)

# Confusion Matrix Without Normalization:
print('Confusion Matrix')
print("=====================================")

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred)

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DT.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

# Confusion Matrix With Normalization:
print('Normalized Confusion Matrix')

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred,normalize='true')

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DT.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

print("=====================================")
print(confusion_matrix(y_test , pred))
print("=====================================")
print(classification_report(y_test , pred))
print("=====================================")
print(accuracy_score(y_test,pred))
print("=====================================")


from sklearn.naive_bayes import GaussianNB
# Making the object of GNB :
GNB = GaussianNB()

# Fitting The Model :
GNB.fit(X_train , y_train)

# Prediciting on test set :
pred = GNB.predict(X_test)

# Confusion Matrix Without Normalization:
print('Confusion Matrix')
print("=====================================")

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred)

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GNB.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

# Confusion Matrix With Normalization:
print('Normalized Confusion Matrix')

# Compute confusion matrix with normalization
cm = confusion_matrix(y_test, pred,normalize='true')

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GNB.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

print("=====================================")
print(confusion_matrix(y_test , pred))
print("=====================================")
print(classification_report(y_test , pred))
print("=====================================")
print(accuracy_score(y_test,pred))
print("=====================================")