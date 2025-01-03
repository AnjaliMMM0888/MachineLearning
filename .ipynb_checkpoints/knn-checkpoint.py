{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "unexpected indent (<ipython-input-1-4f7021350367>, line 77)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-1-4f7021350367>\"\u001b[1;36m, line \u001b[1;32m77\u001b[0m\n\u001b[1;33m    try:\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mIndentationError\u001b[0m\u001b[1;31m:\u001b[0m unexpected indent\n"
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
    "    print(\"KNN classification accuracy\", accuracy(y_test, predictions))\n",
    "    \n",
    "import inspect\n",
    "try:\n",
    "    inspect.getsource(<object>)\n",
    "except OSError as e:\n",
    "    print(f\"Failed to get source for {<object>}: {e}\")\n"
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
