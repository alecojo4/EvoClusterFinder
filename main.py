import time
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


class Logger:
    """
    **printing every message will slow down the execution because print is a costly operation
    **the main role of this class is to save every message in a list because adding to list takes less time than printing
    **after counting the time, print the messages in order
    """

    def _init_(self):
        self.logs = list()

    def log(self, message):
        self.logs.append(message)

    def print(self):
        for message in self.logs:
            print(message)


class City:
    """
    (x,y)->coordinates on 2d axis
    """

    def _init_(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def _repr_(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    """
    returns the distance of a route
    """

    def _init_(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def Average(lst):
    sum = 0
    for i in lst:
        sum = sum + i[1]
    return sum


def __nextGeneration(currentGen, eliteSize, mutationRate, localMax):
    popRanked = rankRoutes(currentGen)

    nextGenerationMax = localMax
    populationAverage = Average(popRanked)

    if populationAverage > localMax:
        nextGenerationMax = populationAverage

    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)

    return (nextGeneration, nextGenerationMax, nextGenerationMax == localMax)


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)

    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    Logger.log("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
   # print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    __localMax = -1
    decrement = False

    for i in range(0, generations):
        if decrement and len(pop) > eliteSize:
            for newIndex in range(int(popSize / generations / 2)):
                pop.pop()
                popSize = popSize - 1
            #Logger.log("Population decremented to "+str(popSize))
        __generationalData = __nextGeneration(pop, eliteSize, mutationRate, __localMax)
        pop = __generationalData[0]
        __localMax = __generationalData[1]
        decrement = __generationalData[2]
        #decrement = False# -> algorithm will work like the classic one

    Logger.log("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    #print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    decrement = False
    __localMax = -1
    for i in range(0, generations):
        if decrement and len(pop) > eliteSize:
            for newIndex in range(int(popSize / generations / 2)):
                pop.pop()
                popSize = popSize - 1
           # Logger.log("Population decremented to " + str(popSize))
        __generationalData = __nextGeneration(pop, eliteSize, mutationRate, __localMax)
        pop = __generationalData[0]
        __localMax = __generationalData[1]
        decrement = __generationalData[2]
        # decrement = False -> algorithm will work like the classic one
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


cityList = []
#25 cities random generated dataset
dataSet = [
    (4, 41)
    ,
    (48, 45)
    ,
    (40, 44)
    ,
    (41, 47)
    ,
    (29, 6)
    ,
    (41, 10)
    ,
    (30, 24)
    ,
    (46, 11)
    ,
    (21, 1)
    ,
    (40, 16)
    ,
    (13, 42)
    ,
    (1, 2)
    ,
    (13, 11)
    ,
    (38, 38)
    ,
    (16, 6)
    ,
    (42, 22)
    ,
    (33, 9)
    ,
    (16, 36)
    ,
    (22, 42)
    ,
    (3, 35)
    ,
    (36, 5)
    ,
    (5, 19)
    ,
    (29, 48)
    ,
    (36, 28)
    ,
    (31, 0)
]

Logger = Logger()

start = time.time()
for i in dataSet:
    cityList.append(City(i[0], i[1]))

geneticAlgorithm(population=cityList, popSize=400, eliteSize=20, mutationRate=0.03, generations=50)
end = time.time()

#print("Performance is ", end - start)
Logger.log("Performance is: "+ str(end - start))
Logger.print()
geneticAlgorithmPlot(population=cityList, popSize=400, eliteSize=20, mutationRate=0.03, generations=50)