from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.keys())

X, y = mnist["data"], mnist["target"]

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
p = svm_clf.predict([some_digit])
print(p)
some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)

d = np.argmax(some_digit_scores)
print(d)

c = svm_clf.classes_
print(c)

c = svm_clf.classes_[5]
print(c)
