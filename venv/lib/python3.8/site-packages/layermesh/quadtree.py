"""Quadtrees for spatial searching in 2D meshes."""

"""Copyright 2021 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

class quadtree(object):
    """Quadtree for spatial searching of mesh columns. On creation, the
    quadtree's bounding box, elements and optional parent quadtree are
    specified.

    Adapted from the quadtree data structure in PyTOUGH.

    """

    def __init__(self, bounds, elements, parent = None):

        from layermesh.geometry import in_rectangle, sub_rectangles

        self.parent = parent #: The quadtree's parent quadtree.
        self.bounds = bounds #: The quadtree's bounding box.
        self.elements = elements #: The elements in the quadtree.
        self.child = [] #: A list of the quadtree's child quadtrees.
        if self.parent:
            #: The generation index of the quadtree.
            self.generation = self.parent.generation + 1
            #: The elements list of the zeroth-generation quadtree.
            self.all_elements = self.parent.all_elements
        else:
            self.generation = 0
            self.all_elements = set(elements)
        if self.num_elements > 1:
            rects = sub_rectangles(self.bounds)
            rect_elements = [[], [], [], []]
            for elt in self.elements:
                for irect, rect in enumerate(rects):
                    if in_rectangle(elt.centre, rect):
                        rect_elements[irect].append(elt)
                        break
            for rect, elts in zip(rects, rect_elements):
                if len(elts) > 0: self.child.append(quadtree(rect, elts, self))

    def __repr__(self): return self.bounds.__repr__()

    def _get_num_elements(self): return len(self.elements)
    #: Number of elements in the quadtree.
    num_elements = property(_get_num_elements)

    def _get_num_children(self): return len(self.child)
    #: Number of children of the quadtree.
    num_children = property(_get_num_children)

    def search_wave(self, pos):
        """Executes search wave for specified point *pos* on a quadtree
        leaf."""

        from layermesh.geometry import rectangles_intersect
        from copy import copy

        todo = copy(self.elements)
        done = []
        while len(todo) > 0:
            elt = todo.pop(0)
            if elt.find(pos): return elt
            done.append(elt)
            for nbr in elt.neighbour & self.all_elements:
                if rectangles_intersect(nbr.bounding_box, self.bounds) and \
                   not ((nbr in done) or (nbr in todo)):
                    todo.append(nbr)
        return None

    def search(self, pos):
        """Returns the element containing the specified point *pos*."""
        leaf = self.leaf(pos)
        if leaf: return leaf.search_wave(pos)
        else: return None

    def leaf(self, pos):
        """Returns the leaf containing the specified point *pos*."""
        from layermesh.geometry import in_rectangle
        if in_rectangle(pos, self.bounds):
            for child in self.child:
                childleaf = child.leaf(pos)
                if childleaf: return childleaf
            return self
        else: return None

    def translate(self, shift):
        """Translates the quadtree horizontally by the specified shift
        vector (an array of length 2).
        """
        self.bounds = [p + shift for p in self.bounds]
        for child in self.child: child.translate(shift)
