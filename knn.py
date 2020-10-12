import math
import os
import shutil
import pandas
from matplotlib import pyplot as plt


def getNormalizeRatio(ratio):
    return (ratio - 0.01) / 0.99


def getNormalizeValue(value):
    return (value - 1.0) / 299


def EuclideanMetric(node1, node2):
    a = (getNormalizeRatio(node1[1]) - getNormalizeRatio(node2[1])) ** 2
    b = (getNormalizeValue(node1[2]) - getNormalizeValue(node2[2])) ** 2
    c = 0 if node1[3] == node2[3] else 1
    return math.sqrt(a + b + c)


def ManhattanMetric(node1, node2):
    a = abs(getNormalizeRatio(node1[1]) - getNormalizeRatio(node2[1]))
    b = abs(getNormalizeValue(node1[2]) - getNormalizeValue(node2[2]))
    c = 0 if node1[3] == node2[3] else 1
    return a + b + c


def ChebyshevMetric(node1, node2):
    a = abs(getNormalizeRatio(node1[1]) - getNormalizeRatio(node2[1]))
    b = abs(getNormalizeValue(node1[2]) - getNormalizeValue(node2[2]))
    c = 0 if node1[3] == node2[3] else 1
    return max(a, b, c)


def printGraph(dataframe, dataframeUndefined, isVote, metricName):
    dataframeTlt = dataframe[dataframe.city == "Тольятти"]
    dataframeSmr = dataframe[dataframe.city == "Самара"]
    dataframeChp = dataframe[dataframe.city == "Чапаевск"]

    fig = plt.figure()
    title = "Метод голосования " if isVote else "Метод, основанный на расстоянии"
    title += " (%s)" % metricName
    fig.suptitle(title)

    plt.scatter(dataframeTlt["value"], dataframeTlt["ratio"],
                c=dataframeTlt["c"].map({1: "blue", 2: "orange", 3: "red"}),
                marker="o")
    plt.scatter(dataframeSmr["value"], dataframeSmr["ratio"],
                c=dataframeSmr["c"].map({1: "blue", 2: "orange", 3: "red"}),
                marker="s")
    plt.scatter(dataframeChp["value"], dataframeChp["ratio"],
                c=dataframeChp["c"].map({1: "blue", 2: "orange", 3: "red"}),
                marker="D")

    ax = plt.gca()
    for i in range(dataframeUndefined.shape[0]):
        node = dataframeUndefined.iloc[i]
        color = {
            1: "blue",
            2: "orange",
            3: "red"
        }[node["c"]]
        plt.scatter(node["value"], node["ratio"], c=color, marker="*", s=150)
        ax.annotate(node["name"], (node["value"], node["ratio"]))

    global nGraph
    nGraph += 1
    plt.savefig(os.path.join("result", "%i.png" % nGraph))
    plt.close(fig)


def proceedKNNVotes(dataframe, dataframeUndefined, metric):
    result = []
    for obj in dataframeUndefined.values:
        distance = []
        for node in dataframe.values:
            distance.append([node[0], node[4], metric(node, obj)])  # ["name" "class" "metric]
        distance.sort(key=lambda x: x[2])

        table = []
        for i in range(dataframe.shape[0]):
            df1 = pandas.DataFrame({"c": [x[1] for x in distance[:i + 1]]})
            c1 = (df1.c == 1).sum()
            c2 = (df1.c == 2).sum()
            c3 = (df1.c == 3).sum()
            if c1 > c2 and c1 > c3:
                res = 1
            elif c2 > c1 and c2 > c3:
                res = 2
            elif c3 > c1 and c3 > c2:
                res = 3
            else:
                res = -1
            table.append(res)

        result.append(pandas.Series(table).mode().values[0])  # number of class
    dataframeUndefined["c"] = result
    printGraph(dataframe, dataframeUndefined, True, metric.__name__)


def proceedKNNDistance(dataframe, dataframeUndefined, metric):
    result = []
    for obj in dataframeUndefined.values:
        distance = []
        for node in dataframe.values:
            distance.append([node[0], node[4], metric(node, obj)])  # ["name" "class" "metric]
        distance.sort(key=lambda x: x[2])

        table = []
        for i in range(dataframe.shape[0]):
            df1 = pandas.DataFrame({
                "c": [x[1] for x in distance[:i + 1]],
                "metric": [x[2] for x in distance[:i + 1]]
            })
            metric1 = df1[df1.c == 1]["metric"].values
            metric2 = df1[df1.c == 2]["metric"].values
            metric3 = df1[df1.c == 3]["metric"].values

            Q1 = 0
            Q2 = 0
            Q3 = 0

            for m in metric1:
                Q1 += 1 / (m ** 2)
            for m in metric2:
                Q2 += 1 / (m ** 2)
            for m in metric3:
                Q3 += 1 / (m ** 2)

            if Q1 > Q2 and Q1 > Q3:
                res = 1
            elif Q2 > Q1 and Q2 > Q3:
                res = 2
            elif Q3 > Q1 and Q3 > Q2:
                res = 3
            else:
                res = -1
            table.append(res)

        result.append(pandas.Series(table).mode().values[0])  # number of class
    dataframeUndefined["c"] = result
    printGraph(dataframe, dataframeUndefined, False, metric.__name__)


df = pandas.read_csv("data.csv")
dfUndefined = pandas.read_csv("undefined.csv")

nGraph = 0

if os.path.isdir("./result"):
    shutil.rmtree("./result")
os.makedirs("./result")

proceedKNNVotes(df, dfUndefined, EuclideanMetric)
proceedKNNDistance(df, dfUndefined, EuclideanMetric)
