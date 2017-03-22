import numpy as np
from math import atan2, fabs, sqrt, floor
from collections import namedtuple

Axis = namedtuple('Axis', ['point', 'direction', 'cost'])
Segment = namedtuple('Segment', ['start', 'end'])

def microsymmetry_axis(P11, P12, P21, P22):
    '''
    Given two pairs of segments starting and ending points, compute 
    microsymmetry reflection axis along with a mean square error for that

    Returns starting point of mycrosymmetry axis Q, direction S, points P11 and P12 reflected
    through axis, endpoint of mycrosymmetry axis R and C2, the mean square error
    Based on:
    Hideo Ogawa. 1991. Symmetry analysis of line drawings using the Hough transform.
    Pattern Recogn. Lett. 12, 1 (January 1991), 9-12. DOI=http://dx.doi.org/10.1016/0167-8655(91)90022-E
    '''
    V1 = P22 - P12
    V2 = P21 - P11
    V = P12 - P11
    B = np.outer(V2, V2) + np.outer(V1, V1)
    A = V1/4.0 - V2/4.0 + V/2.0
    (la, lb), (Ea, Eb) = np.linalg.eig(B - 8 * np.matmul(A, A))
    candidates = []
    for l2, E in ((la, Ea), (lb, Eb)):
        l1 = 8*np.matmul(A, E)
        n = np.matmul(E, V2)+0.25*l1
        S = np.array([-E[1], E[0]])
        Q = P11 + 0.5*n*E
        m = n-2*np.matmul(V, E)
        P11ref = P11 + n*E
        P12ref = P12 + m*E
        e1sq = np.linalg.norm(P12ref-P22) + np.linalg.norm(P11ref-P21)
        e2sq = np.linalg.norm(P12ref-P21) + np.linalg.norm(P11ref-P22)
        C2 = e1sq + e2sq
        candidates.append((Axis(Q, S, C2), P11ref, P12ref))

    result1 = candidates[0]
    result2 = candidates[1]
    return result1 if result1[0].cost < result2[0].cost else result2


def hough_transform(axis):
    '''
    Given axis; returns hough transform point 
    of the axis
    '''
    Sperp = np.array([-axis.direction[1], axis.direction[0]])
    rho = fabs(np.dot(axis.point, Sperp))
    R = Sperp*np.dot(axis.point, Sperp)
    return rho, atan2(R[1], R[0])

def inverse_hough_transform(rho, theta, cost):
    '''
    Given rho and theta, returns an axis
    '''
    direction = np.array((-np.sin(theta), np.cos(theta)), dtype=np.float)
    point = rho * np.array((np.cos(theta), np.sin(theta)), dtype=np.float)
    return Axis(point, direction, cost)

def accumulate_transforms(axis_list, alpha=10, bins=(100, 100), range=None):
    transformed_points = map(hough_transform, axis_list)
    try:
        return np.histogram2d(*zip(*transformed_points), bins=bins, range=range, weights=[1.0/(1+axis.cost*alpha) for axis in axis_list])
    except TypeError:
        print transformed_points
        raise


def equal_segments(segmentA, segmentB):
    return np.array_equal(segmentA.start, segmentB.start) and np.array_equal(segmentA.end, segmentB.end)


def split_segment(segment, length):
    split_segment = segment.end-segment.start
    r = np.linalg.norm(split_segment, ord=1)/length
    split_segment = split_segment/r

    points_to_split = int(floor(r))
    split_points = [segment.start + split_segment*i for i in range(points_to_split)]
    if not split_points or not np.array_equal(split_points[-1], segment.end):
        split_points.append(segment.end)

    return [Segment(split_points[i], split_points[i+1]) for i in range(len(split_points)-1)]

def get_local_maxima(H, threshold=1, neighborhood_size=5):
    import scipy.ndimage.filters as filters
    data_max = filters.maximum_filter(H, neighborhood_size)
    maxima = (H == data_max)
    data_min = filters.minimum_filter(H, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    return np.where(maxima)
 

def find_microsymmetries(segment_list, segment_split_length=0.1):
    axis_list = []
    for sA in segment_list:
        for sB in segment_list:
            if equal_segments(sA, sB):
                continue
            for ssA1, ssA2 in split_segment(sA, segment_split_length):
                for ssB1, ssB2 in split_segment(sB, segment_split_length):
                    axis_list.append(microsymmetry_axis(ssA1, ssA2, ssB1, ssB2)[0])
    
    if axis_list:
        aggregated_hugh, rho_edges, theta_edges = accumulate_transforms(axis_list, range=[[0, np.sqrt(2)], [-np.pi/2, np.pi/2]],)
        threshold = np.amax(aggregated_hugh) * 0.8
        rho_maxs, theta_maxs = get_local_maxima(aggregated_hugh, threshold=threshold)
        maxima = [((rho_edges[rho_i]+rho_edges[rho_i+1])/2, (theta_edges[theta_i]+theta_edges[theta_i+1])/2, aggregated_hugh[rho_i][theta_i]) 
            for rho_i, theta_i in zip(rho_maxs, theta_maxs)]
        
    # print "Found {} maxima".format(len(maxima))
    # for rho, theta, strength in maxima:
    #     print "Maximum at {}, {} with strength {}".format(rho, theta, strength)

        return [inverse_hough_transform(rho, theta, strength) for rho, theta, strength in maxima]
    return []


def debug_microsymmetries():
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    A1, A2, B1, B2 = [0, 0], [0.5, 1], [0.5, -0.5], [1.5, 0]
    S1 = Segment(np.array(A1, dtype = np.float), np.array(A2, dtype = np.float))
    S2 = Segment(np.array(B1, dtype = np.float), np.array(B2, dtype = np.float))

    maxima = find_microsymmetries([S1, S2], segment_split_length = 0.2)

def debug_microsymmetry_axis():
    import cairocffi as cairo
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 200)
    context = cairo.Context(surface)
    context.translate(100*0.1, 100*0.1)
    context.scale(100*0.8, 100*0.8)
    context.scale(1, -1)
    context.translate(0, -1)
    context.set_line_width(context.device_to_user_distance(3, 3)[0])
    A1, A2, B1, B2 = [0, 0], [0.5, 1], [0.5, -0.5], [1.5, 0]
    context.move_to(*A1)
    context.line_to(*A2)
    context.move_to(*B1)
    context.line_to(*B2)
    context.stroke()

    axis, P11ref, P12ref = microsymmetry_axis(*[np.array(v, dtype = np.float) for v in [A1, A2, B1, B2]])

    context.set_source_rgb(1, 0, 0)
    context.move_to(*A1)
    context.line_to(*axis.point)
    context.stroke()
    context.set_source_rgb(0, 1, 0)
    context.move_to(*A1)
    context.line_to(*(A1+axis.direction))
    context.stroke()
    context.set_source_rgb(0, 0, 1)
    context.move_to(*P11ref)
    context.line_to(*P12ref)
    context.stroke()

    surface.write_to_png("microsym.png")
    print axis.cost
    print hough_transform(axis)

if __name__ == '__main__':
    debug_microsymmetries()