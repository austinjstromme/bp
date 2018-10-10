from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    b_main()

def b_main():
    alphas = [2, 5, 10]
    betas = [0.05, 0.2, 0.4]
    cnt = 0

    beta = .1
    for alpha in alphas:
        adj, edges, phi, psi, vals = make_image_model(alpha, beta)
        marg = get_marginals(adj, edges, phi, psi, vals, 50)
    
        x = np.zeros((30, 30))
        for a in range(0, 30):
            for b in range(0, 30):
                x[a, b] = marg[a*30 + b][1]
    
        print(x)
    
        plt.title("alpha, beta = " + str(alpha) + ", " + str(beta))
        plt.imshow(x, cmap='gray')
        plt.savefig('denoise_img_alpha' + str(cnt) + '.png')
        cnt += 1

    alpha = 2
    for beta in betas:
        adj, edges, phi, psi, vals = make_image_model(alpha, beta)
        marg = get_marginals(adj, edges, phi, psi, vals, 50)
    
        x = np.zeros((30, 30))
        for a in range(0, 30):
            for b in range(0, 30):
                x[a, b] = marg[a*30 + b][1]
    
        print(x)
    
        plt.title("alpha, beta = " + str(alpha) + ", " + str(beta))
        plt.imshow(x, cmap='gray')
        plt.savefig('denoise_img_alpha' + str(cnt) + '.png')
        cnt += 1


def a_main():
    """
    alpha = 2
    adj, edges, phi, psi, vals = make_family_tree_a_ii(alpha, 0.8)
    marg = get_marginals(adj, edges, phi, psi, vals, 11)

    print("with alpha = " + str(alpha) + " get marginals:")
    print(str([marg[i][1] for i in range(0, len(adj))]))
    """

    out = ""
    for i in range(1, 12):
        out += " & $p_{x_{" + str(i) + "}}(1)$"
    out += "\\\\ \\hline"
    print(out)

    alphas = [2, 4, 6, 8, 10]
    for alpha in alphas:
        adj, edges, phi, psi, vals = make_family_tree_a_ii(alpha, 0.9)
        marg = get_marginals(adj, edges, phi, psi, vals, 11)

        #print("with alpha = " + str(alpha) + " get marginals:")
        out = str("$\\alpha = " + str(alpha) + "$")
        for i in range(0, len(adj)):
            out += " & " + str("{0:.2f}".format(marg[i][1]))
        out += "\\\\ \\hline"
        print(out)


def make_image_model(alpha, beta):
    im = Image.open("black-white-small-noisy.bmp")
    p = np.array(im)

    n = p.shape[0]
    adj = [[] for i in range(0, n*n)]
    edges = [[] for e in range(0, 2*n*n - 2*n)]

    for i in range(0, n*n):
        if (i % n < n - 1):
            adj[i].append(i - int(i/n))
            edges[i - int(i/n)].append(i)
        if (i % n > 0):
            adj[i].append(i - int(i/n) - 1)
            edges[i - int(i/n) - 1].append(i)
        if (i < n*(n - 1)):
            adj[i].append(n*(n - 1) + i)
            edges[n*(n - 1) + i].append(i)
        if (i > n - 1):
            adj[i].append(n*(n - 2) + i)
            edges[n*(n - 2) + i].append(i)

    phi = [[1., 1.] for i in range(0, n*n)]
    psi = [[alpha, 1., 1., alpha] for e in range(0, len(edges))]

    for i in range(0, n*n):
        if p[int(i/n), i%n] > 0:
            phi[i][0] = beta
            phi[i][1] = 1. - beta
        else:
            phi[i][0] = 1. - beta
            phi[i][1] = beta

    vals = 2

    return [adj, edges, phi, psi, vals]

def make_family_tree_a_i_uncond(alpha):
    adj = [[0], [1], [2], [0, 1, 2, 3], [3, 4], [4, 5], [5, 6, 7], [6, 8, 9], [7], [8], [9]]
    edges = [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [7, 9], [7, 10]]

    phi = [[1., 1.] for i in range(0, len(adj))]
    psi = [[alpha, 1., 1., alpha] for e in range(0, len(edges))]

    vals = 2

    return [adj, edges, phi, psi, vals]

def make_family_tree_a_i_cond(alpha):
    adj = [[0], [1], [2], [0, 1, 2, 3], [3, 4], [4, 5], [5, 6, 7], [6, 8, 9], [7], [8], [9]]
    edges = [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [7, 9], [7, 10]]

    phi = [[1., 1.] for i in range(0, len(adj))]
    psi = [[alpha, 1., 1., alpha] for e in range(0, len(edges))]

    phi[0] = [0., 1.]
    phi[1] = [1., 0.]
    phi[2] = [1., 0.]
    phi[9] = [0., 1.]
    phi[10] = [0., 1.]

    vals = 2

    return [adj, edges, phi, psi, vals]

def make_family_tree_a_ii(alpha, beta):
    adj = [[0], [1], [2], [0, 1, 2, 3], [3, 4], [4, 5], [5, 6, 7], [6, 8, 9], [7], [8], [9]]
    edges = [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [7, 9], [7, 10]]

    phi = [[1., 1.] for i in range(0, len(adj))]
    psi = [[alpha, 1., 1., alpha] for e in range(0, len(edges))]

    phi[0] = [1. - beta, beta]
    phi[1] = [beta, 1. - beta]
    phi[2] = [beta, 1. - beta]
    phi[9] = [1. - beta, beta]
    phi[10] = [1. - beta, beta]

    vals = 2

    return [adj, edges, phi, psi, vals]


def get_marginals(adj, edges, phi, psi, vals, rnds):
    n = len(adj)
    m = sum_prod(adj, edges, phi, psi, vals, rnds)

    marg = [[np.log(phi[i][x_i]) for x_i in range(0, vals)] for i in range(0, n)]

    for i in range(0, n):
        for x_i in range(0, vals):
            for f in adj[i]:
                a = edges[f][0]
                b = edges[f][1]

                if a == i:
                    marg[i][x_i] += m[f + len(edges)][x_i]
                elif b == i:
                    marg[i][x_i] += m[f][x_i]
                else:
                    print("graph not loaded properly")

        s = 0.
        for x_i in range(0, vals):
            print(np.exp(marg[i][x_i]))
            s += np.exp(marg[i][x_i])
        for x_i in range(0, vals):
            marg[i][x_i] = np.exp(marg[i][x_i])/s

    return marg


def sum_prod(adj, edges, phi, psi, vals, rnds):
    # initialize messages IN LOG DOMAIN:
    m = [[0. for x_j in range(0, vals)] for e in range(0, 2*len(edges))]

    # do rnds rounds of parallel sum-product
    for rd in range(0, rnds):
        m = sum_prod_rnd(adj, edges, phi, psi, m, vals)

        #print("after rnd " + str(rd) + " messages are")
        #print(str([[np.exp(m[e][x_i]) for x_i in range(0, vals)] for e in range(0, 2*len(edges))]))

    return m

def sum_prod_rnd(adj, edges, phi, psi, m, vals):
    """
    We assume:
        adj is a list of size n (number of nodes) such that adj[i] is
            the set of all edges with i as an endpoint
        edges is a list such that edges[e] is a list [i, j] where the edge
            is between i and j
        phi is a list of size n such that phi[x] is a list the same size as
            the state space
        psi is a list of size len(edges) such that psi[e]
            is a list of size twice the state space (thought of as
            psi[e][x_i, x_j] = psi[e][x_i + len(vals)*x_j])
        m is a list of size 2*len(edges) such that m[e] is a function from
        vals is number of values in the state space

    Returns a new list m of messages
    """

    n = len(adj)
    mn = [vals for i in range(0, 2*len(edges))]

    for e in range(0, len(edges)):
        i = edges[e][0]
        j = edges[e][1]

        mn[e] = update_messages(i, j, e, adj, edges, phi, psi, m, vals)
        mn[e + len(edges)] = update_messages(j, i, e, adj, edges, phi, psi, m, vals)

    return mn

def update_messages(i, j, e, adj, edges, phi, psi, m, vals):
    msg = [-np.inf for x_j in range(0, vals)]

    for x_j in range(0, vals):
        res = 0.
        for x_i in range(0, vals):
            logprod = 0.

            for f in adj[i]:
                a = edges[f][0]
                b = edges[f][1]

                if a == i and b != j:
                    logprod += m[f + len(edges)][x_i]
                if a != j and b == i:
                    logprod += m[f][x_i]

            res += np.exp(np.log(phi[i][x_i]) + np.log(psi[e][x_i + vals*x_j]) + logprod)

        msg[x_j] = np.log(res)

    delta = np.log(sum([np.exp(msg[x_j]) for x_j in range(0, vals)]))
    for x_j in range(0, vals):
        msg[x_j] -= delta

    return msg

if __name__ == '__main__':
    main()
