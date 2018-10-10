import numpy as np


def main():
    # need to load in data

    # initialize data

def sum_prod(adj, edges, phi, psi, m):
    ""
    We assume:
        adj is a list of size n (number of nodes) such that adj[i] is
            the set of all edges with i as an endpoint
        edges is a list such that edges[e] is a list [i, j] where the edge
            is between i and j
        phi is a list of size n such that phi[n] is a function from state
            space to reals
        psi is a list of size len(edges) such that psi[e]
            is a function from 2-tuples of the state space to reals
        m is a list of size 2*len(edges) such that m[e] is a function from

    Returns a new list m of messages
    ""

    n = len(adj)
    mn = [0 for i in range(0, 2*len(edges))]

    for e in range(0, 2*len(edges)):
        mn[e] = sum([


if __name__ == '__main__':
    main()
