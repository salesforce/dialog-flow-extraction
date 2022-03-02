# Adpated from https://github.com/Renovamen/pcalg-py, which is a python implementation of pcalg.
import itertools
from itertools import combinations, chain
import os
import math
import argparse
import json

from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger


def subset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def skeleton(suffStat, indepTest, alpha, labels, m_max):
    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)):
        G[i][i] = False

    done = False  # done flag

    ord = 0
    n_edgetests = {0: 0}
    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True
        ind = []

        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    ind.append((i, j))

        G1 = G.copy()

        for x, y in ind:
            if G[x][y] == True:
                neighborsBool = [row[x] for row in G1]
                neighborsBool[y] = False

                # adj(C,x) \ {y}
                neighbors = [
                    i for i in range(len(neighborsBool))
                    if neighborsBool[i] == True
                ]

                if len(neighbors) >= ord:
                    # |adj(C, x) \ {y}| > ord
                    if len(neighbors) > ord:
                        done = False

                    for neighbors_S in set(
                            itertools.combinations(neighbors, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        pval = indepTest(suffStat, x, y, list(neighbors_S))

                        if pval >= alpha:
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = list(neighbors_S)
                            break
        ord += 1

    return {'sk': np.array(G), 'sepset': sepset}


def extend_cpdag(graph):
    def rule1(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (
                        search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and (
                        (a, b, c) in graph['unfTriples'] or
                        (c, b, a) in graph['unfTriples']):
                        # if unfaithful, skip
                        continue
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    def rule2(pdag, solve_conf=False):
        search_pdag = pdag.copy()
        ind = []

        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (
                        search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (
                        search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)

            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        # unfaithful
                        if 'unfTriples' in graph.keys() and (
                            (c1, a, c2) in graph['unfTriples'] or
                            (c2, a, c1) in graph['unfTriples']):
                            continue
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    pdag = [[
        0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))
    ] for j in range(len(graph['sk']))]

    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))

    for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
        allZ = []
        for z in range(len(pdag)):
            if graph['sk'][y][z] == True and z != x:
                allZ.append(z)

        for z in allZ:
            if graph['sk'][x][z] == False \
                and graph['sepset'][x][z] != None \
                and graph['sepset'][z][x] != None \
                and not (y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    return np.array(pdag)


def pc(suffStat, alpha, labels, indepTest, m_max=float("inf"), verbose=False):
    graphDict = skeleton(suffStat, indepTest, alpha, labels, m_max)
    cpdag = extend_cpdag(graphDict)
    if verbose:
        print(cpdag)
    return cpdag


def gauss_ci_test(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]

    cut_at = 0.9999999

    if len(S) == 0:
        r = C[x, y]
        # print(r)

    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt(
            (1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))
    else:
        m = C[np.ix_([x] + [y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1 * cut_at, r))

    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))

    if 2 * (1 - norm.cdf(abs(res))) >= 0.05:
        logger.debug(f"{slots[x], slots[y]}")
        # logger.debug(len(S))
        logger.debug(r)

    return 2 * (1 - norm.cdf(abs(res)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_dir',
                        default="dialog-flow-extraction/data/MultiWOZ_2.1",
                        required=False,
                        help="MultiWOZ dialog data directory path")
    parser.add_argument('--domain',
                        type=str,
                        default='hotel',
                        help="MultiWOZ domain to detect causality")

    args = parser.parse_args()

    # Get data
    states = []
    with open(os.path.join(args.data_dir, "data_single.json"), "r") as f:
        data = json.load(f)
        domain = args.domain
        logger.warning(f"Domain: {domain}")

        global slots
        slots = list(data[domain][0]["state"][0].keys())

        logger.warning(f"Slots: {slots}")

        for dialog in data[domain]:
            for turn in dialog["state"]:
                state = [turn[slot][1] for slot in slots]
                states.append(state)
        logger.info(f"#states: {len(states)}")
        logger.debug(f"states: {states}")
    df = pd.DataFrame(states, columns=slots)
    logger.info(f"correlation: {df.corr().values}")

    graph = pc(suffStat={
        "C": df.corr().values,
        "n": len(states)
    },
               alpha=0.05,
               labels=slots,
               indepTest=gauss_ci_test,
               verbose=True)

    G = nx.DiGraph()
    for i in range(len(graph)):
        G.add_node(slots[i])
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                G.add_edges_from([(slots[i], slots[j])])
    nx.draw(G, with_labels=True)
    plt.savefig(f"dialog-flow-extraction/image/{domain}_causality.png")
    plt.show()
