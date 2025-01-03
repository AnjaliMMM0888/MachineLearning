{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
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
     "ename": "NameError",
     "evalue": "name 'plt' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-3-e0c8d6d23590>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     59\u001b[0m     \u001b[0mprint\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     60\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 61\u001b[1;33m     \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     62\u001b[0m     \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcmap\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcmap\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0medgecolor\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m'k'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     63\u001b[0m     \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'plt' is not defined"
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
    "        import matplotlib.pyplot as plt\n",
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
    "    inspect.getsource(main)\n",
    "except OSError as e:\n",
    "    print(f\"Failed to get source for {main}: {e}\")\n"
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
