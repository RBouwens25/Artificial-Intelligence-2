"""kohonen.py"""
import math
import random

class Cluster:
    """This class represents the clusters, it contains the
    prototype and a set with the ID's (which are Integer objects) 
    of the datapoints that are member of that cluster."""
    def __init__(self, dim):
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
        print(len(self.clusters[0][0].prototype))
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
                ## calculate the distances to each prototype for each datapoint
                distances = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
                for row in range(self.n):
                    for col in range(self.n):
                        clus = self.clusters[row][col]
                        distance_squared = 0.0
                        for k in range(self.dim):
                            distance_squared += (datapoint[k] - clus.prototype[k]) ** 2
                        ##add the euclidean distance to the 2d list
                        distances[row][col] = math.sqrt(distance_squared)
                print(distances)
                ##find the BMU, by seeking the minimal distance
                minimal = min([min(r) for r in distances])
                BMU = self.clusters[0][0]
                for row1 in range(self.n):
                    for col1 in range(self.n):
                        if minimal == distances[row1][col1]:
                            BMU = self.clusters[row1][col1]
                            break
                for row2 in range(self.n):
                    for col2 in range(self.n):
                        curcol = self.clusters[row2][col2].prototype
                        for i in range(self.n):
                            ## TODO: this should have multiple if-statements, but i am not sure whether i am doing it correctly.
                            ## It seems to me that you first need to make a list of nodes, and then change them. But also,
                            ## i don't know if it is correct to change all the numbers in the vector.
                            ## I'm afraid I did something wrong in the beginning :(
                            if BMU.prototype[i] + r >= curcol[i]:
                                self.clusters[row2][col2].prototype[i] = (1 - e) * self.clusters[row2][col2].prototype[i] + e * datapoint[i]
        pass

    def test(self):
        # iterate along all clients
        # for each client find the cluster of which it is a member
        # get the actual test data (the vector) of this client
        # iterate along all dimensions
        # and count prefetched htmls
        # count number of hits
        # count number of requests
        # set the global variables hitrate and accuracy to their appropriate value
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