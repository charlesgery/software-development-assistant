#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


BASE_NODE_SIZE = 1e-2
BASE_EDGE_WIDTH = 1e-2


def get_fruchterman_reingold_layout(edge_list,
                                    k                   = None,
                                    scale               = None,
                                    origin              = None,
                                    initial_temperature = 1.,
                                    total_iterations    = 50,
                                    node_size           = None,
                                    node_positions      = None,
                                    fixed_nodes         = None,
                                    *args, **kwargs
):
    """
    Arguments:
    ----------

    edge_list : m-long iterable of 2-tuples or equivalent (such as (m, 2) ndarray)
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    origin : (float x, float y) tuple or None (default None -> (0, 0))
        The lower left hand corner of the bounding box specifying the extent of the layout.
        If None is given, the origin is placed at (0, 0).

    scale : (float delta x, float delta y) or None (default None -> (1, 1))
        The width and height of the bounding box specifying the extent of the layout.
        If None is given, the scale is set to (1, 1).

    k : float or None (default None)
        Expected mean edge length. If None, initialized to the sqrt(area / total nodes).

    total_iterations : int (default 50)
        Number of iterations.

    initial_temperature: float (default 1.)
        Temperature controls the maximum node displacement on each iteration.
        Temperature is decreased on each iteration to eventually force the algorithm
        into a particular solution. The size of the initial temperature determines how
        quickly that happens. Values should be much smaller than the values of `scale`.

    node_size : scalar or (n,) or dict key : float (default 0.)
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
        NOTE: Value is rescaled by BASE_NODE_SIZE (1e-2) to give comparable results to layout routines in igraph and networkx.

    node_positions : dict key : (float, float) or None (default None)
        Mapping of nodes to their (initial) x,y positions. If None are given,
        nodes are initially placed randomly within the bounding box defined by `origin`
        and `scale`.

    fixed_nodes : list of nodes
        Nodes to keep fixed at their initial positions.

    Returns:
    --------
    node_positions : dict key : (float, float)
        Mapping of nodes to (x,y) positions

    """

    # This is just a wrapper around `_fruchterman_reingold` (which implements (the loop body of) the algorithm proper).
    # This wrapper handles the initialization of variables to their defaults (if not explicitely provided),
    # and checks inputs for self-consistency.

    if origin is None:
        if node_positions:
            minima = np.min(list(node_positions.values()), axis=0)
            origin = np.min(np.stack([minima, np.zeros_like(minima)], axis=0), axis=0)
        else:
            origin = np.zeros((2))
    else:
        # ensure that it is an array
        origin = np.array(origin)

    if scale is None:
        if node_positions:
            delta = np.array(list(node_positions.values())) - origin[np.newaxis, :]
            maxima = np.max(delta, axis=0)
            scale = np.max(np.stack([maxima, np.ones_like(maxima)], axis=0), axis=0)
        else:
            scale = np.ones((2))
    else:
        # ensure that it is an array
        scale = np.array(scale)

    assert len(origin) == len(scale), \
        "Arguments `origin` (d={}) and `scale` (d={}) need to have the same number of dimensions!".format(len(origin), len(scale))
    dimensionality = len(origin)

    unique_nodes = _get_unique_nodes(edge_list)
    total_nodes = len(unique_nodes)

    if node_positions is None: # assign random starting positions to all nodes
        node_positions_as_array = np.random.rand(total_nodes, dimensionality) * scale + origin
    else:
        # 1) check input dimensionality
        dimensionality_node_positions = np.array(list(node_positions.values())).shape[1]
        assert dimensionality_node_positions == dimensionality, \
            "The dimensionality of values of `node_positions` (d={}) must match the dimensionality of `origin`/ `scale` (d={})!".format(dimensionality_node_positions, dimensionality)

        is_valid = _is_within_bbox(list(node_positions.values()), origin=origin, scale=scale)
        if not np.all(is_valid):
            error_message = "Some given node positions are not within the data range specified by `origin` and `scale`!"
            error_message += "\nOrigin : {}, {}".format(*origin)
            error_message += "\nScale  : {}, {}".format(*scale)
            for ii, (node, position) in enumerate(node_positions.items()):
                if not is_valid[ii]:
                    error_message += "\n{} : {}".format(node, position)
            raise ValueError(error_message)

        # 2) handle discrepancies in nodes listed in node_positions and nodes extracted from edge_list
        if set(node_positions.keys()) == set(unique_nodes):
            # all starting positions are given;
            # no superfluous nodes in node_positions;
            # nothing left to do
            pass
        else:
            # some node positions are provided, but not all
            for node in unique_nodes:
                if not (node in node_positions):
                    warnings.warn("Position of node {} not provided. Initializing to random position within frame.".format(node))
                    node_positions[node] = np.random.rand(2) * scale + origin

            # unconnected_nodes = []
            for node in node_positions:
                if not (node in unique_nodes):
                    # unconnected_nodes.append(node)
                    warnings.warn("Node {} appears to be unconnected. No position is computed for this node.".format(node))
                    del node_positions[node]

        node_positions_as_array = np.array(list(node_positions.values()))

    if node_size is None:
        node_size = np.zeros((total_nodes))
    elif isinstance(node_size, (int, float)):
        node_size = BASE_NODE_SIZE * node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        node_size = np.array([BASE_NODE_SIZE * node_size[node] if node in node_size else 0. for node in unique_nodes])

    if fixed_nodes is None:
        is_mobile = np.ones((len(unique_nodes)), dtype=np.bool)
    else:
        is_mobile = np.array([False if node in fixed_nodes else True for node in unique_nodes], dtype=np.bool)

    adjacency = _edge_list_to_adjacency_matrix(edge_list)

    # Forces in FR are symmetric.
    # Hence we need to ensure that the adjacency matrix is also symmetric.
    adjacency = adjacency + adjacency.transpose()

    if k is None:
        area = np.product(scale)
        k = np.sqrt(area / float(total_nodes))

    temperatures = _get_temperature_decay(initial_temperature, total_iterations)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # main loop

    for ii, temperature in enumerate(temperatures):
        node_positions_as_array[is_mobile] = _fruchterman_reingold(adjacency, node_positions_as_array,
                                                                   origin      = origin,
                                                                   scale       = scale,
                                                                   temperature = temperature,
                                                                   k           = k,
                                                                   node_radii  = node_size,
        )[is_mobile]

    node_positions_as_array =  _rescale_to_frame(node_positions_as_array, origin, scale)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # format output
    node_positions = dict(zip(unique_nodes, node_positions_as_array))

    return node_positions


def _is_within_bbox(points, origin, scale):
    return np.all((points >= origin) * (points <= origin + scale), axis=1)


def _get_temperature_decay(initial_temperature, total_iterations, mode='quadratic', eps=1e-9):

    x = np.linspace(0., 1., total_iterations)
    if mode == 'quadratic':
        y = (x - 1.)**2 + eps
    elif mode == 'linear':
        y = (1. - x) + eps
    else:
        raise ValueError("Argument `mode` one of: 'linear', 'quadratic'.")

    return initial_temperature * y


def _fruchterman_reingold(adjacency, node_positions, origin, scale, temperature, k, node_radii):
    """
    Inner loop of Fruchterman-Reingold layout algorithm.
    """

    # compute distances and unit vectors between nodes
    delta        = node_positions[None, :, ...] - node_positions[:, None, ...]
    distance     = np.linalg.norm(delta, axis=-1)

    # assert np.sum(distance==0) - np.trace(distance==0) > 0, "No two node positions can be the same!"

    # alternatively: (hack adapted from igraph)
    if np.sum(distance==0) - np.trace(distance==0) > 0: # i.e. if off-diagonal entries in distance are zero
        warning.warn("Some nodes have the same position; repulsion between the nodes is undefined.")
        rand_delta = np.random.rand(*delta.shape) * 1e-9
        is_zero = distance <= 0
        delta[is_zero] = rand_delta[is_zeros]
        distance = np.linalg.norm(delta, axis=-1)

    # subtract node radii from distances to prevent nodes from overlapping
    distance -= node_radii[None, :] + node_radii[:, None]

    # prevent distances from becoming less than zero due to overlap of nodes
    distance[distance <= 0.] = 1e-10 # 1e-13 is numerical accuracy, and we will be taking the square shortly

    with np.errstate(divide='ignore', invalid='ignore'):
        direction = delta / distance[..., None] # i.e. the unit vector

    # calculate forces
    repulsion    = _get_fr_repulsion(distance, direction, k)
    attraction   = _get_fr_attraction(distance, direction, adjacency, k)
    displacement = attraction + repulsion

    # limit maximum displacement using temperature
    displacement_length = np.linalg.norm(displacement, axis=-1)
    displacement = displacement / displacement_length[:, None] * np.clip(displacement_length, None, temperature)[:, None]

    node_positions = node_positions + displacement

    return node_positions


def _get_fr_repulsion(distance, direction, k):
    with np.errstate(divide='ignore', invalid='ignore'):
        magnitude = k**2 / distance
    vectors   = direction * magnitude[..., None]
    # Note that we cannot apply the usual strategy of summing the array
    # along either axis and subtracting the trace,
    # as the diagonal of `direction` is np.nan, and any sum or difference of
    # NaNs is just another NaN.
    # Also we do not want to ignore NaNs by using np.nansum, as then we would
    # potentially mask the existence of off-diagonal zero distances.
    vectors   = _set_diagonal(vectors, 0)
    return np.sum(vectors, axis=0)


def _get_fr_attraction(distance, direction, adjacency, k):
    magnitude = 1./k * distance**2 * adjacency
    vectors   = -direction * magnitude[..., None] # NB: the minus!
    vectors   = _set_diagonal(vectors, 0)
    return np.sum(vectors, axis=0)


def _rescale_to_frame(node_positions, origin, scale):
    node_positions = node_positions.copy() # force copy, as otherwise the `fixed_nodes` argument is effectively ignored
    node_positions -= np.min(node_positions, axis=0)
    node_positions /= np.max(node_positions, axis=0)
    node_positions *= scale[None, ...]
    node_positions += origin[None, ...]
    return node_positions


def _set_diagonal(square_matrix, value=0):
    n = len(square_matrix)
    is_diagonal = np.diag(np.ones((n), dtype=np.bool))
    square_matrix[is_diagonal] = value
    return square_matrix


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _get_unique_nodes(edge_list):
    """
    Using numpy.unique promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.
    """
    return list(set(_flatten(edge_list)))


def _edge_list_to_adjacency_matrix(edge_list, edge_weights=None):

    sources = [s for (s, _) in edge_list]
    targets = [t for (_, t) in edge_list]
    if edge_weights:
        weights = [edge_weights[edge] for edge in edge_list]
    else:
        weights = np.ones((len(edge_list)))

    # map nodes to consecutive integers
    nodes = sources + targets
    unique = set(nodes)
    indices = range(len(unique))
    node_to_idx = dict(zip(unique, indices))

    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    adjacency_matrix[source_indices, target_indices] = weights

    return adjacency_matrix


if __name__ == '__main__':

    import networkx as nx

    # create a graph
    n = 10 # number of nodes
    G = nx.complete_graph(n)
    edge_list = list(G.edges())

    # compute a "spring" layout that takes node sizes into account
    node_size = dict(zip(range(n), np.arange(0, 100, 10))) # dict : node ID -> node size
    node_positions = get_fruchterman_reingold_layout(edge_list, node_size=node_size, k=0.01)

    nx.draw(G, pos=node_positions, node_size=[300*node_size[node] for node in node_positions]); plt.show()