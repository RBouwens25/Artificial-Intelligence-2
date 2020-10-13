"""kohonen.py"""
import math
import random

class Cluster:
    """This class represents the clusters, it contains the
    prototype and a set with the ID's (which are Integer objects) 
    of the datapoints that are member of that cluster."""
    def __init__(self, dim):
        ##Initialize cluster with random values
        self.prototype = [random.random() for _ in range(dim)]
        self.current_members = set()

class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(dim) for _ in range(n)] for _ in range(n)]

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0


    def train(self):
        # Step 1: initialize map with random vectors (A good place to do this, is in the initialisation of the clusters)
        # Repeat 'epochs' times:
        #     Step 2: Calculate the square size and the learning rate, these decrease linearly with the number of epochs.
        #     Step 3: Every input vector is presented to the map (always in the same order)
        #     For each vector its Best Matching Unit is found, and :
        #         Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.
        # Since training kohonen maps can take quite a while, presenting the user with a progress bar would be nice

        ## repeat everything in every epoch
        for i in range(self.epochs):
            ## calculate size of the neighbourhood
            r = (self.n/2)*(1 - i/self.epochs)
            ## calculate learningrate
            e = 0.8 * (1- i/self.epochs)
            ##step3
            for j, datapoint in enumerate(self.traindata):
                ## calculate the distances to each node for the datapoint
                distances = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
                for row in range(self.n):
                    for col in range(self.n):
                        node = self.clusters[row][col]
                        distance_squared = 0.0
                        for k in range(self.dim):
                            distance_squared += (datapoint[k] - node.prototype[k]) ** 2
                        ##add the euclidean distance to the 2d list
                        distances[row][col] = math.sqrt(distance_squared)
                ##find the BMU, by seeking the minimal distance
                minimal = min([min(r) for r in distances])
                bmu = None
                for row in range(self.n):
                    for col in range(self.n):
                        if minimal == distances[row][col]:
                            bmu = self.clusters[row][col]
                            break
                    if bmu is not None:
                        break

                ##Learning step
                eta = self.initial_learning_rate * (1 - i/self.epochs)
                for row in range(self.n):
                    for col in range(self.n):
                        node = self.clusters[row][col]
                        distance_squared = 0.0
                        for k in range(self.dim):
                            distance_squared += (bmu.prototype[k] - node.prototype[k]) ** 2
                        ## if the node is within the neighbourhood
                        if(math.sqrt(distance_squared) <= r):
                           ## Adjust node to be more like input vector
                           for k in range(self.dim):
                               old = node.prototype[k]
                               self.clusters[row][col].prototype[k] = (1-eta) * old + eta * datapoint[k]


    def test(self):
        # iterate along all clients
        prefetched = 0
        hits = 0
        requests = 0
        for i, client in enumerate(self.traindata):
            # for each client find the cluster of which it is a member
            distances = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
            for row in range(self.n):
                for col in range(self.n):
                    node = self.clusters[row][col]
                    distance_squared = 0.0
                    for k in range(self.dim):
                        distance_squared += (client[k] - node.prototype[k]) ** 2
                    ##add the euclidean distance to the 2d list
                    distances[row][col] = math.sqrt(distance_squared)
            ##find the BMU, by seeking the minimal distance
            minimal = min([min(r) for r in distances])
            bmu = None
            for row in range(self.n):
                for col in range(self.n):
                    if minimal == distances[row][col]:
                        bmu = self.clusters[row][col]
                        break
                if bmu is not None:
                    break
            # get the actual test data (the vector) of this client
            # iterate along all dimensions
            for d in range(self.dim):
                # and count prefetched htmls
                if bmu.prototype[d] >= self.prefetch_threshold:
                    prefetched += 1
                    # count number of hits
                    if client[d]:
                        hits += 1
                # count number of requests
                if client[d]:
                    requests += 1
        # set the global variables hitrate and accuracy to their appropriate value
        self.accuracy = requests / prefetched 
        self.hitrate = hits / requests 
        pass

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster["+str(i)+"]["+str(j)+"] :", self.clusters[i][j].current_members)
                print()

    def print_prototypes(self):
        for i in range(self.n):
            for j in range(self.n):
               print("Prototype cluster["+str(i)+"]["+str(j)+"] :", self.clusters[i][j].prototype)
               print()
