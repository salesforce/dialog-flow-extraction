import collections
import json
import time
from itertools import takewhile

import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
import numpy as np
from loguru import logger

import params


def draw_split(trans_freq, edge_visited, domain=None, save_path=None):
    G = nx.DiGraph()
    node_labels = {}
    for i in range(len(trans_freq)):
        G.add_node(i)
        node_labels[i] = i

    for i in range(len(trans_freq)):
        for j in range(len(trans_freq)):
            if trans_freq[i, j] > 0.0:
                G.add_edge(i, j)

    node_visited = np.zeros((len(trans_freq), 3))
    for i in range(len(trans_freq)):
        for j in range(len(trans_freq)):
            for split in range(3):
                if edge_visited[(i, j)][split]:
                    node_visited[i, split] = 1
                    node_visited[j, split] = 1
    logger.debug(f"node_visited: {node_visited}")

    # node_color = np.zeros(len(trans_freq))
    node_color = []
    cnt = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(trans_freq)):
        if np.sum(node_visited[i, :]) == 0:
            node_color.append('w')
            cnt[0] += 1
        elif np.sum(node_visited[i, :]) == 1:
            if node_visited[i, 0] == 1:
                node_color.append('r')  # train
                cnt[1] += 1
            elif node_visited[i, 1] == 1:
                node_color.append('g')  # val
                cnt[2] += 1
            else:
                node_color.append('b')  # test
                cnt[3] += 1
        elif np.sum(node_visited[i, :]) == 2:
            if node_visited[i, 0] == 0:
                node_color.append('m')
                cnt[4] += 1
            elif node_visited[i, 1] == 0:
                node_color.append('c')
                cnt[5] += 1
            else:
                node_color.append('y')
                cnt[6] += 1
        else:
            node_color.append('k')
            cnt[7] += 1
    logger.warning(f"-----Node overlap-----")
    logger.warning(
        f"Train only: {cnt[1]}, Val only: {cnt[2]}, Test only: {cnt[3]}")
    # logger.warning(
    #     f"Train: {cnt[1] + cnt[5] + cnt[6] + cnt[7]}, Val: {cnt[2] + cnt[4] + cnt[6] + cnt[7]}, Test: {cnt[3] + cnt[4] + cnt[5] + cnt[7]}"
    # )
    logger.warning(
        f"Train&Val: {cnt[6] + cnt[7]}, Train&Test: {cnt[5] + cnt[7]}, Val&Test: {cnt[4] + cnt[7]}"
    )
    logger.warning(f"Train&Val&Test: {cnt[7]}")

    cnt_edge = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(trans_freq)):
        for j in range(len(trans_freq)):
            for split in range(3):
                if edge_visited[(i, j)][split]:
                    cnt_edge[split] += 1
            if edge_visited[(i, j)][0] and edge_visited[(i, j)][1]:
                cnt_edge[3] += 1  # train&val
            if edge_visited[(i, j)][0] and edge_visited[(i, j)][2]:
                cnt_edge[4] += 1  # train&test
            if edge_visited[(i, j)][1] and edge_visited[(i, j)][2]:
                cnt_edge[5] += 1  # val&test
            if edge_visited[(i, j)][0] and edge_visited[(
                    i, j)][1] and edge_visited[(i, j)][2]:
                cnt_edge[6] += 1  # train&val&test
    logger.warning(f"-----Edge overlap-----")
    logger.warning(
        f"Train only: {cnt_edge[0] - cnt_edge[3] - cnt_edge[4] + cnt_edge[6]}, Val: {cnt_edge[1] - cnt_edge[3] - cnt_edge[5] + cnt_edge[6]}, Test: {cnt_edge[2] - cnt_edge[4] - cnt_edge[5] + cnt_edge[6]}"
    )
    logger.warning(
        f"Train&Val: {cnt_edge[3]}, Train&Test: {cnt_edge[4]}, Val&Test: {cnt_edge[5]}"
    )
    logger.warning(f"Train&Val&Test: {cnt_edge[6]}")

    logger.debug(node_color)
    logger.debug(len(node_color))

    pos = nx.kamada_kawai_layout(G)
    draw_networkx_nodes_ellipses(G,
                                 pos=pos,
                                 node_width=20,
                                 node_height=20,
                                 node_color=node_color,
                                 edge_color='k',
                                 alpha=1.0)
    # nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    plt.axis('off')
    import matplotlib.patches as mpatches
    w_patch = mpatches.Patch(color='w', label='Invalid Nodes')
    r_patch = mpatches.Patch(color='r', label='Train Only')
    g_patch = mpatches.Patch(color='g', label='Valid Only')
    b_patch = mpatches.Patch(color='b', label='Test Only')
    m_patch = mpatches.Patch(color='m', label='Valid&Test')
    c_patch = mpatches.Patch(color='c', label='Train&Test')
    y_patch = mpatches.Patch(color='y', label='Valid&Test')
    k_patch = mpatches.Patch(color='k', label='Train&Valid&Test')

    plt.legend(handles=[
        r_patch, g_patch, b_patch, m_patch, c_patch, y_patch, k_patch, w_patch
    ],
               loc="lower left")

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.suptitle(domain.capitalize(), fontsize=20)
    if save_path:
        fig.savefig(save_path)
    plt.clf()


def detect_community_and_draw(trans_freq, k=10, domain=None, save_path=None):
    G = nx.DiGraph()
    node_labels = {}
    for i in range(len(trans_freq)):
        G.add_node(i)
        node_labels[i] = i

    for i in range(len(trans_freq)):
        for j in range(len(trans_freq)):
            if trans_freq[i, j] > 0.0:
                G.add_edge(i, j)

    start_time = time.time()
    k = 10
    comp = girvan_newman(G)
    logger.debug(
        f"Community detection computation time: {time.time() - start_time}")
    limited = takewhile(lambda c: len(c) <= k, comp)
    for communities in limited:
        last_community = tuple(sorted(c) for c in communities)
        # logger.debug(last_community)

    node_color = np.zeros(len(trans_freq))
    for i, com in enumerate(last_community):
        for node in com:
            node_color[node] = i
    # logger.debug(node_color)

    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G)
    draw_networkx_nodes_ellipses(G,
                                 pos=pos,
                                 node_width=20,
                                 node_height=20,
                                 node_color=node_color,
                                 edge_color='k',
                                 alpha=1.0,
                                 cmap='rainbow')
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.suptitle(domain, fontsize=20)
    if save_path:
        fig.savefig(save_path)
    plt.clf()

    return last_community, node_color


def detect_cycle(adj_matrix, self_loop=False):
    # detect cycle in the direct graph by DFS
    visited = [False] * (len(adj_matrix))
    rec_stack = [False] * (len(adj_matrix))
    for node in range(len(adj_matrix)):
        if visited[node] == False:
            if is_cycle(adj_matrix,
                        node,
                        visited,
                        rec_stack,
                        self_loop=self_loop) == True:
                return True
    return False


def is_cycle(adj_matrix, v, visited, rec_stack, self_loop=False):
    # Mark current node as visited and adds to recursion stack
    visited[v] = True
    rec_stack[v] = True

    # Recur for all neighbours if any neighbour is visited and in rec_stack then graph is cyclic
    for neighbour in np.nonzero(adj_matrix[v])[0]:
        if neighbour == v and not self_loop:
            continue
        if visited[neighbour] == False:
            if is_cycle(adj_matrix, neighbour, visited, rec_stack) == True:
                return True
        elif rec_stack[neighbour] == True:
            return True

    # The node needs to be poped from recursion stack before function ends
    rec_stack[v] = False
    return False


def get_slots(slot_description_file):
    # read slot names for each domain
    domain_slot_dict = {}
    for d in params.domain:
        domain_slot_dict[d] = []
    with open(slot_description_file, "r") as f:
        slots = json.load(f).keys()
        for s in slots:
            domain = s.split("-")[0]
            if "book " in s:
                slot = s.replace("book ", "book-")
            else:
                slot = s.replace("-", "-semi-")
            if domain in domain_slot_dict:
                domain_slot_dict[domain].append(slot)
    all_slots = []
    for d in params.domain:
        all_slots.extend(domain_slot_dict[d])

    return domain_slot_dict, all_slots


def draw_networkx_nodes_ellipses(G,
                                 pos,
                                 nodelist=None,
                                 node_height=1,
                                 node_width=2,
                                 node_angle=0,
                                 node_color='r',
                                 edge_color='b',
                                 node_shape='o',
                                 alpha=1.0,
                                 cmap=None,
                                 vmin=None,
                                 vmax=None,
                                 ax=None,
                                 linewidths=None,
                                 label=None,
                                 **kwds):

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
        return None

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise nx.NetworkXError('Node %s has no position.' % e)
    except ValueError:
        raise nx.NetworkXError('Bad value in node positions.')

    if isinstance(alpha, collections.Iterable):
        logger.debug("Apply alpha to nodes")
        node_color = nx.apply_alpha(node_color, alpha, nodelist, cmap, vmin,
                                    vmax)
        alpha = None

    if cmap is not None:
        cm = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cm = norm = None

    node_collection = mpl.collections.EllipseCollection(
        widths=node_width,
        heights=node_height,
        angles=0,
        offsets=np.array(xy),
        cmap=cm,
        norm=norm,
        transOffset=ax.transData,
        linewidths=linewidths)

    node_collection.set_alpha(alpha)
    node_collection.set_label(label)
    if isinstance(node_color, list) or isinstance(node_color, str):
        node_collection.set_facecolor(node_color)
    else:
        node_color = np.array(list(node_color))
        node_collection.set_array(node_color)
    node_collection.set_edgecolor(edge_color)
    node_collection.set_zorder(2)
    ax.add_collection(node_collection)
    ax.autoscale_view()

    return node_collection


def visualize(trans_freq, states, threshold=0.1, save_path=None):
    G = nx.DiGraph()
    node_labels = {}
    for i in range(len(trans_freq)):
        G.add_node(i)
        node_labels[i] = i

    # edge_labels = {}
    for i in range(len(trans_freq)):
        for j in range(len(trans_freq)):
            if trans_freq[i, j] > threshold:
                G.add_edge(i, j)
                # edge_labels[(i, j)] = "%.2f" % trans_freq[i, j]

    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G)
    draw_networkx_nodes_ellipses(G,
                                 pos=pos,
                                 node_width=20,
                                 node_height=20,
                                 node_color='w',
                                 edge_color='k',
                                 alpha=1.0)

    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    # plt.show()
    if save_path:
        fig.savefig(save_path)
    plt.clf()


def draw_generation_results():
    # Perplexity
    plt.rcParams.update({'font.size': 16})

    for domain in params.domain:
        ax = plt.axes(projection='3d')
        ax.set_xlabel('\ntrain ratio', linespacing=1)
        ax.set_ylabel('\naugment ratio', linespacing=1)
        ax.set_zlabel('\nPPL ', linespacing=1)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_zticks(np.arange(5, 45, 5))
        ax.set_zlim(5, 40)
        with open(f"out/generation/results_{domain}_aug.txt", "r") as f:
            f.readline()
            train_ratio = []
            aug_ratio = []
            ppl = []

            for i, line in enumerate(f):
                row = line.split(" ")
                train_ratio.append(float(row[0]))
                aug_ratio.append(float(row[1]))
                ppl.append(float(row[2]))

                if (i + 1) % 5 == 0:
                    ax.plot3D(train_ratio, aug_ratio, ppl, 'red')
                    train_ratio = []
                    aug_ratio = []
                    ppl = []

        with open("out/generation/results_hotel_mfs.txt", "r") as f:
            f.readline()
            train_ratio = []
            aug_ratio = []
            ppl = []
            bleu = []

            for i, line in enumerate(f):
                row = line.split(" ")
                train_ratio.append(float(row[0]))
                aug_ratio.append(float(row[1]))
                ppl.append(float(row[2]))

                if (i + 1) % 5 == 0:
                    ax.plot3D(train_ratio, aug_ratio, ppl, 'blue')
                    train_ratio = []
                    aug_ratio = []
                    ppl = []
        fig = plt.gcf()
        fig.set_size_inches(8, 8)
        fig.suptitle(domain.capitalize(), y=0.88, fontsize=27)
        fig.savefig(f"image/generation_ppl_{domain}.png", bbox_inches='tight')
        plt.clf()
        logger.info(f"Saving figure to image/generation_ppl_{domain}.png")

    # BLEU
    for domain in params.domain:
        ax = plt.axes(projection='3d')
        ax.set_xlabel('\ntrain ratio', linespacing=1)
        ax.set_ylabel('\naugment ratio', linespacing=1)
        ax.set_zlabel('\nBLEU', linespacing=1)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_zticks(np.arange(5, 30, 5))
        ax.set_zlim(5, 25)
        with open(f"out/generation/results_{domain}_aug.txt", "r") as f:
            f.readline()
            train_ratio = []
            aug_ratio = []
            bleu = []

            for i, line in enumerate(f):
                row = line.split(" ")
                train_ratio.append(float(row[0]))
                aug_ratio.append(float(row[1]))
                bleu.append(float(row[3]))

                if (i + 1) % 5 == 0:
                    ax.plot3D(train_ratio, aug_ratio, bleu, 'red')
                    train_ratio = []
                    aug_ratio = []
                    bleu = []

        with open("out/generation/results_hotel_mfs.txt", "r") as f:
            f.readline()
            train_ratio = []
            aug_ratio = []
            bleu = []

            for i, line in enumerate(f):
                row = line.split(" ")
                train_ratio.append(float(row[0]))
                aug_ratio.append(float(row[1]))
                bleu.append(float(row[3]))

                if (i + 1) % 5 == 0:
                    ax.plot3D(train_ratio, aug_ratio, bleu, 'blue')
                    train_ratio = []
                    aug_ratio = []
                    bleu = []
        fig = plt.gcf()
        fig.set_size_inches(8, 8)
        fig.suptitle(domain.capitalize(), y=0.88, fontsize=27)
        fig.savefig(f"image/generation_bleu_{domain}.png", bbox_inches='tight')
        plt.clf()
        logger.info(f"Saving figure to image/generation_bleu_{domain}.png")


if __name__ == '__main__':
    # draw_generation_results()
    draw_generation_results()