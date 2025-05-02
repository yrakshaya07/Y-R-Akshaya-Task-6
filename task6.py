
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(r"C:\Users\AKSHAYA\Downloads\Iris.csv")

feature_cols = ['PetalLengthCm', 'PetalWidthCm']
target_col = 'Species'

X = df[feature_cols].values
y = df[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

k_values = [1, 3, 5, 7]

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nK = {k} | Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix (K={k})')
    plt.show()

    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid_points)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure()
    le = LabelEncoder()
    Z_num = le.fit_transform(Z.ravel())
    Z_num = Z_num.reshape(xx.shape)

    plt.contourf(xx, yy, Z_num, alpha=0.3, cmap=plt.cm.Set1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=pd.factorize(y)[0], edgecolor='k', cmap=plt.cm.Set1)
    plt.title(f"Decision Boundary (K={k})")
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.show()
