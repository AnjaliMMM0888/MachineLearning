{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(120, 4)\n",
      "[5.1 2.5 3.  1.1]\n",
      "(120,)\n",
      "[1 2 0 2 1 0 0 0 0 1 0 1 0 2 2 0 2 2 2 2 0 2 2 1 1 1 1 1 1 0 0 2 2 2 0 0 0\n",
      " 2 1 2 2 1 0 2 0 2 0 1 1 0 1 0 2 2 2 1 0 0 2 1 1 0 1 2 1 1 1 0 0 0 1 1 0 2\n",
      " 1 2 2 1 0 1 2 0 0 2 2 1 1 2 0 1 2 2 2 1 0 0 0 0 2 1 2 0 0 1 1 2 1 1 2 2 2\n",
      " 0 2 0 0 2 2 1 0 0]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN classification accuracy 1.0\n"
     ]
    }
   ],
   "source": [
    "from collections import Counter\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "def euclidean_distance(x1, x2):\n",
    "    return np.sqrt(np.sum((x1 - x2) ** 2))\n",
    "\n",
    "\n",
    "class KNN:\n",
    "    def __init__(self, k=3):\n",
    "        self.k = k\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        self.X_train = X\n",
    "        self.y_train = y\n",
    "\n",
    "    def predict(self, X):\n",
    "        y_pred = [self._predict(x) for x in X]\n",
    "        return np.array(y_pred)\n",
    "\n",
    "    def _predict(self, x):\n",
    "        # Compute distances between x and all examples in the training set\n",
    "        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]\n",
    "        # Sort by distance and return indices of the first k neighbors\n",
    "        k_idx = np.argsort(distances)[: self.k]\n",
    "        # Extract the labels of the k nearest neighbor training samples\n",
    "        k_neighbor_labels = [self.y_train[i] for i in k_idx]\n",
    "        # return the most common class label\n",
    "        most_common = Counter(k_neighbor_labels).most_common(1)\n",
    "        return most_common[0][0]\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    # Imports\n",
    "    from matplotlib.colors import ListedColormap\n",
    "    from sklearn import datasets\n",
    "    from sklearn.model_selection import train_test_split\n",
    "\n",
    "    cmap = ListedColormap([\"#FF0000\", \"#00FF00\", \"#0000FF\"])\n",
    "\n",
    "    def accuracy(y_true, y_pred):\n",
    "        accuracy = np.sum(y_true == y_pred) / len(y_true)\n",
    "        return accuracy\n",
    "\n",
    "    iris = datasets.load_iris()\n",
    "    X, y = iris.data, iris.target\n",
    "\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X, y, test_size=0.2, random_state=1234\n",
    "    )\n",
    "    \n",
    "    #Training Samples\n",
    "    print (X_train.shape)\n",
    "    print (X_train[0])\n",
    "\n",
    "    # Training labels\n",
    "    print(y_train.shape)\n",
    "    print (y_train)\n",
    "\n",
    "    plt.figure()\n",
    "    plt.scatter(X[:, 0] , X[:, 1], c=y, cmap=cmap, edgecolor ='k', s=20)\n",
    "    plt.show()\n",
    "\n",
    "    a = [1,1,1,2,2,3,4,5,6]\n",
    "    from collections import Counter\n",
    "    most_common = Counter(a).most_common(1)\n",
    "    print(most_common[0][0])\n",
    "\n",
    "    k = 3\n",
    "    clf = KNN(k=k)\n",
    "    clf.fit(X_train, y_train)\n",
    "    predictions = clf.predict(X_test)\n",
    "    print(\"KNN classification accuracy\", accuracy(y_test, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
