#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
from operator import itemgetter
from math import sin, cos, radians, sqrt
from symmetries import find_microsymmetries, Segment
import numpy as np

def setup_canvas(height, width):
    import cairocffi as cairo
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, height, width)
    context = cairo.Context(surface)
    with context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
    context.translate(height*0.1, width*0.1)
    context.scale(height*0.8, width*0.8)
    context.scale(1, -1)
    context.translate(0, -1)

    return surface, context

def draw_segments_to_context(segments, context, frame):
    context.set_line_width(context.device_to_user_distance(3, 3)[0])
    if frame:
        context.move_to(0,0)
        context.line_to(1,0)
        context.line_to(1,1)
        context.line_to(0,1)
        context.line_to(0,0)
        context.stroke()
    
    for a, b in segments:
        context.move_to(*a)
        context.line_to(*b)
        context.stroke()

def draw_segments(segments, filename, frame = True):
    surface, context = setup_canvas(100, 100)
    draw_segments_to_context(segments, context, frame)
    surface.write_to_png(filename)

def draw_in_grid(segments_list, grid_size, filename):
    import cairocffi as cairo
    slot_height = 100
    slot_width = 100
    Nx = grid_size[0]
    Ny = grid_size[1]
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, Nx*slot_height, Ny*slot_width)
    context = cairo.Context(surface)
    with context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
    context.translate(slot_height*0.1, slot_width*0.1)
    context.scale(slot_height*0.8, slot_width*0.8)
    context.scale(1, -1)
    # Arrange a list of list of segments into a grid and outputs it to a file

    for k, segments in enumerate(segments_list):
        i = k % Nx
        j = int(k / Nx)
        with context:
            context.translate(i + i*0.25, -j-1.05 - j*0.2)
            segments = segments_list[i + j*Nx]
            draw_segments_to_context(segments, context, False)

    surface.write_to_png(filename)

def draw_points(points, filename, frame = True):
    from math import pi
    surface, context = setup_canvas(1000, 1000)
    radius = context.device_to_user_distance(2, 2)[0]
    context.set_line_width(radius)
    for point in points:
        context.move_to(*point)
        context.arc(point[0], point[1], radius, 0, 2*pi)
        context.fill()


    surface.write_to_png(filename)

def stroke(size, angle, initial):
    # print "Stroke of length {} at {}º, starting from {}".format(size, angle, initial)
    return initial[0] + size * cos(radians(angle)), initial[1] + size * sin(radians(angle))

def interpolate(a, b, x):
    return tuple(round(i + (j - i) * x, 6) for i, j in zip(a, b))

# Kudos to /u/Kakila for idea of use adjacency list for topology
def parse(glyph):
    from itertools import chain
    def to_size(size_index):
        return glyph['sizes'][size_index]

    def to_angle(direction_index):
        return glyph['directions'][direction_index] * 90

    def connected_subgraphs(graph):
        return nx.connected_component_subgraphs(graph)

    def lengths(g, nodes):
        try:
            return [(i, to_size(graph.node[i]['lengths'])) for i in nodes]
        except KeyError:
            print "OK", graph.node, i
            raise


    g_data = glyph['graph']
    topology = g_data['adjacency']
    roots = glyph['roots']
    edge_list = [(i,j) for i, connected in enumerate(topology) for j in connected]

    #if not edge_list:
    #    print "NO EDGES",  [(i, connected) for i, connected in enumerate(topology)]

    if edge_list:
        #print "BUILDING FROM EDGES", glyph
        graph = nx.Graph(edge_list)
    elif topology:
        #print "BUILDING FROM NODES", glyph
        graph = nx.Graph()
        graph.add_nodes_from(range(len(topology)))

    for k in ('lengths', 'directions', 'positions'):
        try:
            nx.set_node_attributes(graph, k, dict(enumerate(g_data[k])))
        except KeyError:
            print "graph ", edge_list
            print g_data, k
            raise
    
    def draw_strokes_from(main_edges):
        strokes = {}
        segments = []
        for start, cluster_root in zip(main_edges, roots):
            for i,j,k in nx.dfs_labeled_edges(graph, start):
                if k['dir']!='forward':
                    continue
                node_attr = graph.node[j]
                if i in strokes:
                    last_start, last_finish = strokes[i]
                    x = node_attr['positions'] if node_attr['positions'] else 0
                    stroke_start = interpolate(last_start, last_finish, x)
                else:
                    stroke_start = cluster_root
                stroke_end = stroke(to_size(node_attr['lengths']), to_angle(node_attr['directions']), stroke_start)
                strokes[j] = Segment(np.array(stroke_start), np.array(stroke_end))
                segments.append(strokes[j])
                # print [nx.dfs_successors(graph, start) for i in nx.dfs_successors(graph, start)[start]]
                # draw_strokes_from(stroke_start, stroke_end, nx.dfs_predecessors(graph, start))
        return segments
    # Start drawing from the longest node of each connected subgraph
    try:
        return chain(draw_strokes_from([max(lengths(graph, nodes), key = itemgetter(1))[0] for nodes in connected_subgraphs(graph)]))
    except KeyError:
        raise


vertical = 1

original_runes = {
    'fehu': {
        'directions': (vertical, 0.5),
        'sizes': (1, 0.94, 0.55),
        'roots': ((0.3, 0),),
        'graph': { # Each node is a stroke
            'lengths': (0, 1, 2), # stroke lengths
            'directions': (0, 1, 1), # stroke directions
            'adjacency': ((), (0,), (0,)), # stroke adjacency
            'positions': (None, 0.3, 0.6), # connection position    
        }
    },
    'jera': {
        'directions': (0.5, 1.5, 2.5, 3.5),
        'sizes': (0.3, 0.2),
        'roots': ((0.45, 0.25), (0.55, 0.75)),
        'graph': {
            'adjacency': ((), (0,), (), (2,)),
            'lengths': (0, 1, 0, 1),
            'directions': (0, 1, 2, 3),
            'positions': (None, 1, None, 1),    
        }
    },
    'perp': {
        'directions': (1, 0.5, -0.5),
        'sizes': (1, 0.4),
        'roots': ((0.2, 0),),
        'graph': {
            'lengths': (0, 1, 1, 1, 1),
            'directions': (0, 2, 1, 1, 2),
            'adjacency': ((), (0,), (1,), (0,), (3,)),
            'positions': (None, 1, 1, 0, 1),
        }
    }
}

def reproduce(rune, number_of_children):
    from random import gauss
    from math import pi

    size_drift_strength = 0.1
    direction_drift_strength = 0.1

    def clamp(x, smallest, greatest):
        return min(max(x, smallest), greatest)

    def normalize_angle(x):
        while x < 0:
            x += 4

        while x > 4:
            x -= 4

        return x

    def drift_length(x):
        return clamp(gauss(x, size_drift_strength), 0, 1)

    def drift_angle(x):
         return normalize_angle(gauss(x, direction_drift_strength))

    def assign(origin, new_fields):
        ret = origin.copy()
        ret.update(new_fields)
        return ret

    directions = rune['directions']
    sizes = rune['sizes']
    positions = rune['graph']['positions']
    
    children = []
    for i in range(number_of_children):
        new_directions = tuple(drift_angle(angle) for angle in directions)
        new_sizes = tuple(drift_length(x) for x in sizes)
        new_positions = tuple(drift_length(x) if x else None for x in positions)

        keep = dict((k, j) for j, k in enumerate([i for i, length in enumerate(new_sizes) if length>0]))

        updated_graph_elements = {'positions': new_positions}
        if len(keep) == 0:
            continue

        if len(keep) < len(new_sizes):
            print "prune", keep, new_sizes
            new_sizes = tuple(length for length in new_sizes if length > 0)

            

            ls = rune['graph']['lengths']
            ds = rune['graph']['directions']
            ads = rune['graph']['adjacency']
            nps = new_positions
            node_keep = dict((k, j) for j,k in enumerate(i for i, l_i in enumerate(ls) if l_i in keep))

            print ads, node_keep
            new_lengths, new_direction_indexes, new_adjacency, new_positions = zip(*[(keep[l_i], d_i, map(keep.get, ad_i), np_i) for l_i, d_i, ad_i, np_i in zip(ls, ds, ads, nps) if l_i in keep])
            
            new_adjacency = tuple(tuple(node_keep.get(a) for a in adj if a in node_keep) for adj in new_adjacency)

            print new_adjacency
            updated_graph_elements.update({
                'positions': new_positions,
                'lengths': new_lengths,
                'directions': new_direction_indexes,
                'adjacency': new_adjacency
            })
            assert len(new_positions) == len(new_lengths)
            assert len(new_direction_indexes) == len(new_lengths)
            assert len(new_adjacency) == len(new_lengths)
        else:
            assert len(directions) == len(new_directions)
            assert len(new_positions) == len(positions)
            assert len(new_sizes) == len(sizes)
        new_child = assign(rune, {
            'directions': new_directions,
            'sizes': new_sizes,
            'graph': assign(rune['graph'], updated_graph_elements)
        })
        children.append(new_child)

    return children


def to_point_cloud(segments):
    from random import uniform
    from math import sqrt
   
    # Sample segments as points at a rate of 100 points per unit length
    points = []
    for start, end in segments:
        length = sqrt(sum((j - i)**2 for i, j in zip(start, end)))
        samples = int(round(length*100))
        phases = (uniform(0, 1) for i in range(samples))

        points.extend(interpolate(start, end, s) for s in phases)
    return points

def visual_rate(segments):
    symmetries = find_microsymmetries(segments, segment_split_length=0.2)
    return len(symmetries)

def glyph_rate(glyph):
    return np.std(glyph['directions'])
    

def rate(rune):
    return visual_rate(parse(rune))*glyph_rate(rune)

runes = original_runes.values()
generations = 10
CHILDREN_PER_GENERATION = 5
MAX_POOL = 100
for i in range(generations):
    print len(runes)
    children = []
    for data in runes:
        children.extend(reproduce(data, CHILDREN_PER_GENERATION))
    runes = sorted(children, key=rate, reverse=True)[:MAX_POOL]


print map(rate, runes[:10])
print runes[:10]
    #draw_segments(segments, '{}.png'.format(name), frame=False)
    #draw_points(to_point_cloud(segments), '{}_cloud.png'.format(name), frame=True)
draw_in_grid(map(parse, runes), (int(round(sqrt(len(runes)))), int(round(sqrt(len(runes))))), 'runes.png')

