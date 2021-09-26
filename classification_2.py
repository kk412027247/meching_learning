from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.keys())


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


X, y = mnist["data"], mnist["target"]

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

svm_clf = SVC()
# svm_clf.fit(X_train, y_train)
# p = svm_clf.predict([some_digit])
# print(p)
# some_digit_scores = svm_clf.decision_function([some_digit])
# print(some_digit_scores)
#
# d = np.argmax(some_digit_scores)
# print(d)
#
# c = svm_clf.classes_
# print(c)
#
# c = svm_clf.classes_[5]
# print(c)

# ovr_clf = OneVsRestClassifier(SVC())
# ovr_clf.fit(X_train, y_train)
# r = ovr_clf.predict([some_digit])
# print(r)
#
# l = len(ovr_clf.estimators_)
# print(l)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
r = sgd_clf.predict([some_digit])
print(r)

d = sgd_clf.decision_function(([some_digit]))
print(d)

s = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(s)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
s = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
#
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()
#
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
#
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))
plt.subplot(221)
plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222)
plot_digits(X_ab[:25], images_per_row=5)

plt.subplot(223)
plot_digits(X_ba[:25], images_per_row=5)

plt.subplot(224)
plot_digits(X_bb[:25], images_per_row=5)

plt.show()
