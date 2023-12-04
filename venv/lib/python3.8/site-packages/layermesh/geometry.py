"""Geometry calculations."""

"""Copyright 2019 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

def in_polygon(pos, polygon):
    """Tests if the point *pos* (a tuple, list or array of length 2) a
    lies within a given polygon (a tuple or list of points, each
    itself a tuple, list or array of length 2).

    """
    if len(polygon) == 2:
        return in_rectangle(pos, polygon)
    else:
        tolerance = 1.e-6
        numcrossings = 0
        pos = np.array(pos)
        poly = [np.array(v, dtype = float) for v in polygon]
        ref = poly[0]
        v = pos - ref
        for i in range(len(poly)):
            p1 = poly[i] - ref
            i2 = (i+1) % len(poly)
            p2 = poly[i2] - ref
            if p1[1] <= v[1] < p2[1] or p2[1] <= v[1] < p1[1]:
                d = p2 - p1
                if abs(d[1]) > tolerance:
                    x = p1[0] + (v[1] - p1[1]) * d[0] / d[1]
                    if v[0] < x: numcrossings += 1
        return (numcrossings % 2)

def in_rectangle(pos, rect):
    """Tests if the point *pos* lies in an axis-aligned rectangle, defined
    as a two-element tuple or list of points [bottom left, top right],
    each itself a tuple, list or array of length 2.

    """
    return all([rect[0][i] <= pos[i] <= rect[1][i] for i in range(2)])

def rectangles_intersect(rect1, rect2):
    """Returns *True* if two rectangles intersect."""
    return all([(rect1[1][i] >= rect2[0][i]) and
                (rect2[1][i] >= rect1[0][i]) for i in range(2)])

def sub_rectangles(rect):
    """Returns the sub-rectangles formed by subdividing the given
    rectangle evenly in four."""
    centre = 0.5 * (rect[0] + rect[1])
    r0 = [rect[0], centre]
    r1 = [np.array([centre[0], rect[0][1]]), np.array([rect[1][0], centre[1]])]
    r2 = [np.array([rect[0][0], centre[1]]), np.array([centre[0], rect[1][1]])]
    r3 = [centre, rect[1]]
    return [r0, r1, r2, r3]

def bounds_of_points(points):
    """Returns bounding box around the specified tuple, list, array or set
    of points, each one a tuple, list or array of length 2."""
    bottomleft = np.array([min([pos[i] for pos in points]) for i in range(2)])
    topright = np.array([max([pos[i] for pos in points]) for i in range(2)])
    return [bottomleft, topright]

def rect_to_poly(rect):
    """Converts a rectangle to a polygon."""
    return [rect[0], np.array(rect[1][0], rect[0][1]),
            rect[1], np.array(rect[0][0], rect[1][1])]

def polygon_area(polygon):
    """Calculates the (unsigned) area of an arbitrary polygon (a tuple,
    list or array of points, each one a tuple, list or array of length
    2)."""
    area = 0.0
    n = len(polygon)
    if n > 0:
        poly = [np.array(v, dtype = float) for v in polygon]
        poly -= poly[0]
        for j, p1 in enumerate(poly):
            p2 = poly[(j+1) % n]
            area += p1[0] * p2[1] - p2[0] * p1[1]
    return 0.5 * abs(area)

def polygon_centroid(polygon):
    """Calculates the centroid of an arbitrary polygon (a tuple, list or
    array of points, each one a tuple, list or array of length 2).
    """
    c, area = np.zeros(2), 0.
    poly = [np.array(v, dtype = float) for v in polygon]
    n = len(poly)
    shift = poly[0]
    poly -= shift # shift to reduce roundoff for large coordinates
    if n < 3: return sum(poly) / n + shift
    else:
        for j, p1 in enumerate(poly):
            p2 = poly[(j+1) % n]
            t = p1[0] * p2[1] - p2[0] * p1[1]
            area += t
            c += (p1 + p2) * t
        area *= 0.5
        return c / (6. * area) + shift

def line_polygon_intersections(polygon, line, bound_line = (True,True),
                               indices = False):
    """Returns a list of the intersection points at which a line crosses a
    polygon.  The list is sorted by distance from the start of the
    line.  The parameter bound_line controls whether to limit
    intersections between the line's start and end points.  If indices
    is True, also return polygon side indices of intersections.
    """
    from numpy.linalg import LinAlgError
    crossings = []
    poly = [np.array(v, dtype = float) for v in polygon]
    line = [np.array(p) for p in line]
    ref = poly[0]
    l1, l2 = line[0] - ref, line[1] - ref
    tol = 1.e-12
    ind = {}
    def in_unit(x): return -tol <= x <= 1.0 + tol
    for i, p in enumerate(poly):
        p1 = p - ref
        p2 = poly[(i+1)%len(poly)] - ref
        dp = p2 - p1
        A, b = np.column_stack((dp, l1 - l2)), l1 - p1
        try:
            xi = np.linalg.solve(A,b)
            inline = True
            if bound_line[0]: inline = inline and (-tol <= xi[1])
            if bound_line[1]: inline = inline and (xi[1] <= 1.0 + tol)
            if in_unit(xi[0]) and inline :
                c = tuple(ref + p1 + xi[0] * dp)
                ind[c] = i
        except LinAlgError: continue
    crossings = [np.array(c) for c, i in ind.items()]
    # Sort by distance from start of line:
    sortindex = np.argsort([np.linalg.norm(c - line[0]) for c in crossings])
    if indices: return [crossings[i] for i in sortindex], \
       [ind[tuple(crossings[i])] for i in sortindex]
    else: return [crossings[i] for i in sortindex]

def polyline_polygon_intersections(polygon, polyline):
    """Returns a list of intersection points at which a polyline (a tuple,
    list or array of points, each one a tuple, list or array of length
    2) crosses a polygon.
    """
    N = len(polyline)
    intersections = [
        line_polygon_intersections(polygon, [pt, polyline[(i+1) % N]])
        for i, pt in enumerate(polyline)]
    from itertools import chain # flatten list of lists
    return list(chain.from_iterable(intersections))

def simplify_polygon(polygon, tolerance = 1.e-6):
    """Simplifies a polygon by deleting colinear points.  The tolerance
    for detecting colinearity of points can optionally be specified.
    """
    s = []
    poly = [np.array(v, dtype = float) for v in polygon]
    N = len(poly)
    for i, p in enumerate(poly[:]):
        l, n = poly[i-1], poly[(i+1)%N]
        dl, dn = p - l, n - p
        if np.dot(dl, dn) < (1. - tolerance) * \
           np.linalg.norm(dl) * np.linalg.norm(dn):
            s.append(p)
    return s

def polygon_boundary(this, other, polygon):
    """Returns point on a line between vector this and other and also on
    the boundary of the polygon."""
    big = 1.e10
    poly = [np.array(v, dtype = float) for v in polygon]
    ref = poly[0]
    a = this - ref
    b = other - ref
    v = None
    dmin = big
    for i in range(len(poly)):
        c = poly[i] - ref
        i2 = (i + 1) % len(poly)
        d = poly[i2] - ref
        M = np.transpose(np.array([b - a, c - d]))
        try:
            r = np.linalg.solve(M, c - a)
            if r[0] >= 0.0 and 0.0 <= r[1] <= 1.0:
                bdy = c * (1.0 - r[1]) + d * r[1]
                dist = np.linalg.norm(bdy - a)
                if dist < dmin:
                    dmin = dist
                    v = bdy + ref
        except np.linalg.LinAlgError: continue
    return v

def line_projection(a, line, return_xi = False):
    """Finds projection of point *a* onto a *line* (defined by two points,
    each a tuple, list or array of length 2).  Optionally returns the
    non-dimensional distance xi between the line start and end.
    """
    line = [np.array(p) for p in line]
    d = line[1] - line[0]
    try:
        xi = np.dot(a - line[0], d) / np.dot(d, d)
        p = line[0] + d * xi
    except ZeroDivisionError: # line ill-defined
        p, xi = None, None
    if return_xi: return p, xi
    else: return p

def point_line_distance(a, line):
    """Finds the distance between point *a* and a line."""
    return np.linalg.norm(a - line_projection(a, line))

def polyline_line_distance(polyline, line):
    """Returns minimum distance between a polyline and a line."""
    dists = []
    for i, pt in enumerate(polyline):
        pline = [pt, polyline[(i + 1) % len(polyline)]]
        if line_polygon_intersections(line, pline): return 0.0
        else: dists.append(min([point_line_distance(p, line) for p in pline]))
    return min(dists)

def vector_heading(p):
    """Returns heading angle of a point p (tuple, list or array of length
    2), in radians clockwise from the y-axis ('north').
    """
    from math import asin
    p = np.array(p)
    theta = asin(p[0] / np.linalg.norm(p))
    if p[1] < 0: theta = np.pi - theta
    if theta < 0: theta += 2. * np.pi
    elif theta > 2. * np.pi: theta -= 2. * np.pi
    return theta

def rotation(angle, centre = None):
    """Returns 2-by-2 matrix A and vector b representing a rotation of the
    specified angle (degrees clockwise) about the specified centre (or
    the origin if no centre is specified). The rotation of a point p
    is then given by Ap + b.
    """
    from math import radians, sin, cos
    if centre is None: c = np.zeros(2)
    else: c = np.array(centre)
    angleradians = radians(angle)
    cosangle, sinangle = cos(angleradians), sin(angleradians)
    A = np.array([[cosangle, sinangle], [-sinangle, cosangle]])
    b = np.dot(np.identity(2) - A, c)
    return A, b

