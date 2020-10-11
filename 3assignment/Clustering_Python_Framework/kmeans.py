"""kmeans.py"""
import math
import random

class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()

class KMeans:
    def __init__(self, k, traindata, testdata, dim):
        self.k = k
        self.traindata = traindata
        print(self.traindata)
        self.testdata = testdata
        self.dim = dim

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        # An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

        self.seed = 14135

    ##TODO: update prototypes
    ## A function to handle updating prototypes
    def updatePrototypes(self):
        for i,_ in enumerate(self.clusters):
            self.clusters[i].prototype = [0.0 for _ in range(self.dim)]
            for d in range(self.dim):
                dimSum = 0.0
                for m in self.clusters[i].current_members:
                    dimSum += self.traindata[m][d]
                self.clusters[i].prototype[d] = dimSum / len(self.clusters[i].current_members)

    ## A function to handle each step
    def step(self):
        ## move the current members to previous members and empty the set
        for i,_ in enumerate(self.clusters):
            self.clusters[i].previous_members = self.clusters[i].current_members
            self.clusters[i].current_members = set()
        ## calculate the distances to each prototype for each datapoint 
        for m, datapoint in enumerate(self.traindata):
            distances = [0.0 for _ in range(self.k)]
            for c, cluster in enumerate(self.clusters):
                distance_squared = 0.0
                ##sum the squared distance between each point
                for i in range(self.dim):
                    distance_squared += (datapoint[i] - cluster.prototype[i]) ** 2 
                ##add the euclidean distance to the array
                distances[c] = math.sqrt(distance_squared)
            ##select the closest cluster and assign the member to it
            index_min = min(range(len(distances)), key=distances.__getitem__) 
            self.clusters[index_min].current_members.add(m)

    def isUnstable(self):
        stable = True
        for cluster in self.clusters:
            if cluster.current_members != cluster.previous_members:
                stable = False
        return stable

    def train(self):
        # implement k-means algorithm here:
        # Step 1: Select an initial random partioning with k clusters
        random.seed(self.seed)
        for i in range(len(self.traindata)):
            ##Choose a random cluster and assign the member to it
            cluster = random.randrange(self.k)
            self.clusters[cluster].current_members.add(i)
        self.updatePrototypes()
        # Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
        self.step()
        # Step 3: recalculate cluster centers
        self.updatePrototypes()
        # Step 4: repeat until clustermembership stabilizes
        while self.isUnstable():
            self.step()
            self.updatePrototypes()
        pass

    def test(self):
        # iterate along all clients. Assumption: the same clients are in the same order as in the testData
        # for each client find the cluster of which it is a member
        # get the actual testData (the vector) of this client
        # iterate along all dimensions
        # and count prefetched htmls
        # count number of hits
        # count number of requests
        # set the variables hitrate and accuracy to their appropriate value
        pass


    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i, cluster in enumerate(self.clusters):
            print("Members cluster["+str(i)+"] :", cluster.current_members)
            print()

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster["+str(i)+"] :", cluster.prototype)
            print()
