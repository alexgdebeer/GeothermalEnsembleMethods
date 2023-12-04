"""Layered computational meshes."""

"""Copyright 2019 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

default_cell_type_sort = -1 # decreasing

class node(object):
    """2-D mesh node. On creation, the node's horizontal position (and
    optionally index) are specified.
    """

    def __init__(self, pos, index = None):
        self.pos = np.array(pos) #: Array containing the node's horizontal position.
        self.index = index #: Integer containing the node's index in the mesh.
        self.column = set() #: Set containing the column objects the node belongs to.

    def __repr__(self):
        return str(list(self.pos))

    def find(self, polygon, indices = False):
        """Returns self if the node is inside the specified *polygon* (tuple,
        list or array of 2-D points, each one a tuple, list or array
        of length 2), otherwise *None*."""
        from layermesh.geometry import in_polygon
        return self if in_polygon(self.pos, polygon) else None

class _layered_object(object):
    """Object with a list of layers, used as the base class for column and
    mesh."""

    def _find_layer(self, z):
        """Returns layer containing the elevation z, or *None* if the point is
        outside the mesh."""
        if self.num_layers == 0:
            return None
        else: # use bisection to find layer:
            if self.layer[-1].bottom <= z <= self.layer[0].top:
                i0, i1 = 0, self.num_layers - 1
                while i1 > i0:
                    im = (i0 + i1) // 2
                    if z >= self.layer[im].bottom: i1 = im
                    else: i0 = im + 1
                return self.layer[i1]
            else:
                return None

class column(_layered_object):
    """Mesh column. On creation, the column's nodes (and optionally index)
    are specified."""

    def __init__(self, node, index = None):
        self.node = node #: List of the node objects in the column.
        self.index = index #: Integer containing the column's index in the mesh.
        self._centroid = None
        self._area = None
        #: Set containing the neighbouring columns (those that share a face).
        self.neighbour = set()
        #: List of layers in the column.
        self.layer = None
        #: List of cells in the column.
        self.cell = None

    def __repr__(self):
        return str(self.index)

    def _get_num_nodes(self): return len(self.node)
    num_nodes = property(_get_num_nodes) #: Number of nodes in the column.

    def _get_num_layers(self): return len(self.layer)
    num_layers = property(_get_num_layers) #: Number of layers in the column.

    def _get_num_cells(self): return len(self.cell)
    num_cells = property(_get_num_cells) #: Number of cells in the column.

    def _get_num_neighbours(self): return len(self.neighbour)
    #: Number of neighbouring columns (those that share a face).
    num_neighbours = property(_get_num_neighbours)

    def _get_polygon(self):
        return [node.pos for node in self.node]
        #: Polygon (list of arrays of length 2) formed by column node positions.
    polygon = property(_get_polygon)

    def _get_centroid(self):
        if self._centroid is None:
            from layermesh.geometry import polygon_centroid
            self._centroid = polygon_centroid(self.polygon)
        return self._centroid
    #: Column centroid.
    centroid = property(_get_centroid)
    #: Column centroid.
    centre = property(_get_centroid)

    def _get_area(self):
        if self._area is None:
            from layermesh.geometry import polygon_area
            self._area = polygon_area(self.polygon)
        return self._area
    #: Area of column.
    area = property(_get_area)

    def _get_volume(self):
        return self.area * sum([lay.thickness for lay in self.layer])
    #: Column volume.
    volume = property(_get_volume)

    def _get_bounding_box(self):
        from layermesh.geometry import bounds_of_points
        return bounds_of_points([n.pos for n in self.node])
    #: Horizontal bounding box of column.
    bounding_box = property(_get_bounding_box)

    def _get_interior_angle(self):
        side = np.array([n.pos - self.node[i - 1].pos
                         for i, n in enumerate(self.node)])
        costheta = np.array([np.dot(s, side[i - 1]) / \
                             (np.linalg.norm(s) * np.linalg.norm(side[i - 1]))
                             for i, s in enumerate(side)])
        return np.pi - np.arccos(costheta)
    #: Array of interior angles for each node in the column.
    interior_angle = property(_get_interior_angle)

    def _get_angle_ratio(self):
        angles = self.interior_angle
        return np.max(angles) / np.min(angles)
    #: Angle ratio, defined as the ratio of the largest interior angle
    #: to the smallest interior angle.
    #:
    #: This can be used as a measure
    #: of the skewness of the column, with values near 1 being less
    #: skewed.
    angle_ratio = property(_get_angle_ratio)

    def _get_face_length(self):
        return np.array([np.linalg.norm(
            n.pos - self.node[i - 1].pos)
            for i, n in enumerate(self.node)])
    #: Array of lengths of the column faces.
    face_length = property(_get_face_length)

    def _get_face_length_ratio(self):
        l = self.face_length
        return np.max(l) / np.min(l)
    #: Face length ratio, defined as the ratio of the
    #: longest face length to the shortest face length (a
    #: generalisation of the aspect ratio for quadrilateral columns).
    face_length_ratio = property(_get_face_length_ratio)

    def set_layers(self, layers, num_layers):
        """Sets column layers to be the last *num_layers* layers from the
        specified list."""
        istart = len(layers) - num_layers
        self.layer = layers[istart:]

    def set_surface(self, layers, surface = None):
        """Sets column layers from the given list, according to the specified
        surface elevation.

        If *surface* = *None*, then the column is assigned all layers
        in the list. Otherwise, it is assigned all layers with centres
        below the specified surface elevation.

        """
        if surface is None: self.layer = layers
        else:
            self.layer = [lay for lay in layers
                          if lay.centre <= surface]

    def _get_surface(self):
        return self.layer[0].top
    #: Surface elevation of the column, given by the top elevation of its uppermost layer.
    #: (This property is read-only: use **set_surface()** or **set_layers()**
    #: to set the layers in the column.)
    surface = property(_get_surface)

    def _find_column(self, pos):
        """Returns self if the column contains the 2-D point pos (tuple, list or
        array of length 2), otherwise *None*."""
        from layermesh.geometry import in_polygon
        return self if in_polygon(np.array(pos), self.polygon) else None

    def _find_columns(self, polygon):
        """Returns self if the column centroid is inside the
        polygon (a tuple, list or array of 2-D points, each a tuple,
        list or array of length 2).
        """
        from layermesh.geometry import in_polygon
        return self if in_polygon(self.centroid, polygon) else None

    def _find_cell(self, pos):
        """Returns cell in the column containing the 3-D point pos (tuple, list or
        array of length 3), otherwise *None*."""
        lay = self._find_layer(pos[2])
        if lay:
            col = self._find_column(pos[:2])
            if col: return lay.column_cell[col.index]
            else: return None
        else: return None

    def find(self, match, indices = False, sort = False):
        """Returns cells, columns or layers satisfying the criterion *match*.

        The *match* parameter can be:

        * a **function** taking a cell and returning a Boolean: a list
          of matching cells is returned
        * a **scalar**: *match* is interpreted as an **elevation**,
          and the layer containing it is returned
        * a **2-D point** (tuple, list or array of length 2): *match*
          is interpreted as a **horizontal position**, and self is returned
          if it contains the point
        * a **polygon** (tuple, list or array of 2-D points): self is
          returned if its centroid is inside the polygon
        * a **3-D point** (tuple, list or array of length 3): *match*
          is interpreted as a **3-D position**, and the cell
          containing it is returned

        If indices is *True*, the cell, column or layer indices
        are returned rather than the cells, columns or layers
        themselves.

        If *sort* is *True*, then lists of results are sorted by index.

        If no match is found, then *None* is returned, except when the
        expected result is a list, in which case an empty list is
        returned.
        """

        if callable(match):
            result = [c for c in self.cell if match(c)]
        elif isinstance(match, (float, int)):
            result = self._find_layer(match)
        elif isinstance(match, (tuple, list, np.ndarray)):
            pos = np.array(match)
            if pos.ndim == 1:
                if len(pos) == 1:
                    result = self._find_layer(pos[0])
                elif len(pos) == 2:
                    result = self._find_column(pos)
                elif len(pos) == 3:
                    result = self._find_cell(pos)
                else:
                    raise Exception('Length of point to find is not between 1 and 3.')
            elif pos.ndim == 2:
                if pos.shape[1] == 2:
                    result = self._find_columns(pos)
                else:
                    raise Exception('Unrecognised match shape.')
            else:
                raise Exception('Unrecognised match shape.')
        else:
            raise Exception('Unrecognised match type.')

        if result is None: return None
        elif isinstance(result, list):
            if sort:
                isort = np.argsort(np.array([r.index for r in result]))
                result = [result[i] for i in isort]
            return [r.index for r in result] if indices else result
        else:
            return result.index if indices else result

    def translate(self, shift):
        """Translates column horizontally by the specified shift array (a
        tuple, list or array of length 2)."""
        if self._centroid is not None:
            self._centroid += np.array(shift)

    def _get_side_neighbour(self):
        nbrs = []
        for i, nodei in enumerate(self.node):
            i1 = (i + 1) % self.num_nodes
            side_nodes = set([nodei, self.node[i1]])
            nbr = None
            for col in self.neighbour:
                if side_nodes < set(col.node):
                    nbr = col
                    break
            nbrs.append(nbr)
        return nbrs
    #: List of neighbouring columns corresponding to each column
    #: side (or None if the column side is on a boundary).
    side_neighbour = property(_get_side_neighbour)

class column_face(object):
    """Face between two columns. On creation, the two columns on either side
    of the face are specified."""

    def __init__(self, column):
        #: List or tuple of column objects on either side of the face.
        self.column = column
        #: List of node objects at either end of the face.
        self.node = list(set(column[0].node) & set(column[1].node))

    def _get_angle_cosine(self):
        n = self.node[1].pos - self.node[0].pos
        n = n / np.linalg.norm(n)
        d = self.column[1].centre - self.column[0].centre
        d = d / np.linalg.norm(d)
        return np.dot(n, d)
    #: Cosine of angle between the face and the line joining the
    #: column centroids on either side.
    #:
    #: This can be used to measure the orthogonality of the face:
    #: orthogonal faces have angle cosine zero.
    angle_cosine = property(_get_angle_cosine)

class layer(object):
    """Mesh layer. On creation, the bottom and top elevations of the layer
    (and optionally the layer index) are specified."""

    def __init__(self, bottom, top, index = None):
        self.bottom = bottom #: Bottom elevation of the layer.
        self.top = top #: Top elevation of the layer.
        self.index = index #: Layer index in the mesh (numbered from top down).
        self._centre = None
        self.column = None #: List of columns in the layer.
        self.cell = None #: List of cells in the layer.
        self.column_cell = None #: Dictionary of cells, keyed by column indices.
        self.above = None #: Layer above this one, if it exists, otherwise None.
        self.below = None #: Layer below this one, if it exists, otherwise None.
        self._quadtree = None

    def __repr__(self):
        return str(self.index)

    def _get_num_columns(self): return len(self.column)
    #: Number of columns in the layer.
    num_columns = property(_get_num_columns)

    def _get_num_cells(self): return len(self.cell)
    #: Number of cells in the layer.
    num_cells = property(_get_num_cells)

    def _get_node(self):
        nodes = set()
        for col in self.column:
            for n in col.node: nodes.add(n)
        return nodes
    #: Set of nodes in the layer.
    node = property(_get_node)

    def _get_centre(self):
        if self._centre is None:
            self._centre = 0.5 * (self.bottom + self.top)
        return self._centre
    #: Elevation of layer centre.
    centre = property(_get_centre)

    def _get_thickness(self):
        return self.top - self.bottom
    #: Vertical thickness of layer.
    thickness = property(_get_thickness)

    def _get_area(self):
        return sum([col.area for col in self.column])
    #: Horizontal area of layer.
    area = property(_get_area)

    def _get_volume(self):
        return self.area * self.thickness
    #: Volume of layer.
    volume = property(_get_volume)

    def _get_horizontal_bounds(self):
        from layermesh.geometry import bounds_of_points
        return bounds_of_points([n.pos for n in self.node])
    #: Horizontal bounding box for layer (list of two arrays of length 2,
    #: representing the bottom left and top right corner coordinates of the
    #: bounding box).
    horizontal_bounds = property(_get_horizontal_bounds)

    def _get_quadtree(self):
        if self._quadtree is None:
            from layermesh import quadtree
            self._quadtree = quadtree.quadtree(self.horizontal_bounds,
                                               self.column)
        return self._quadtree
    #: Quadtree object for column searching within the layer.
    quadtree = property(_get_quadtree)

    def translate(self, shift):
        """Translates layer by specified 3-D shift (a tuple, list or array of
        length 3)."""
        self.bottom += shift[2]
        self.top += shift[2]
        if self._centre is not None:
            self._centre += shift[2]
        if self._quadtree is not None:
            self._quadtree.translate(shift[:2])

    def _find_layer(self, z):
        """Returns self if the layer contains the specified elevation *z*,
        or None otherwise."""
        return self if self.bottom <= z <= self.top else None

    def _find_column(self, pos):
        """Returns the column in the layer containing the specified 2-D point
        pos (tuple, list or array of length 2), or *None*."""
        return self.quadtree.search(pos)

    def _find_columns(self, polygon):
        """Returns a list of columns in the layer with centroids inside the
        polygon (a tuple, list or array of 2-D points, each a tuple,
        list or array of length 2).
        """
        return [col for col in self.column if col._find_columns(polygon)]

    def _find_cell(self, pos):
        """Returns the cell in the layer containing the specified 3-D point
        pos (tuple, list or array of length 3), or *None*."""
        if self._find_layer(pos[2]):
            col = self._find_column(pos[:2])
            if col: return self.column_cell[col.index]
            else: return None
        else: return None

    def find(self, match, indices = False, sort = False):
        """Returns cells, columns or layer satifying the criterion *match*.

        The *match* parameter can be:

        * a **function** taking a cell and returning a Boolean: a list
          of matching cells is returned
        * a **scalar**: *match* is interpreted as an **elevation**,
          and the layer is returned if the elevation is inside it
        * a **2-D point** (tuple, list or array of length 2): *match*
          is interpreted as a **horizontal position**, and the column
          containing it is returned
        * a **polygon** (tuple, list or array of 2-D points): a list of
          columns inside the polygon are returned
        * a **3-D point** (tuple, list or array of length 3): *match*
          is interpreted as a **3-D position**, and the cell
          containing it is returned

        If indices is *True*, the cell, column or layer indices are
        returned rather than the cells, columns or layer themselves.

        If *sort* is *True*, then lists of results are sorted by index.

        If no match is found, then *None* is returned, except when the
        expected result is a list, in which case an empty list is
        returned.
        """

        if callable(match):
            result = [c for c in self.cell if match(c)]
        elif isinstance(match, (float, int)):
            result = self._find_layer(match)
        elif isinstance(match, (tuple, list, np.ndarray)):
            pos = np.array(match)
            if pos.ndim == 1:
                if len(pos) == 1:
                    result = self._find_layer(pos[0])
                elif len(pos) == 2:
                    result = self._find_column(pos)
                elif len(pos) == 3:
                    result = self._find_cell(pos)
                else:
                    raise Exception('Length of point to find is not between 1 and 3.')
            elif pos.ndim == 2:
                if pos.shape[1] == 2:
                    result = self._find_columns(pos)
                else:
                    raise Exception('Unrecognised match shape.')
            else:
                raise Exception('Unrecognised match shape.')
        else:
            raise Exception('Unrecognised match type.')

        if result is None: return None
        elif isinstance(result, list):
            if sort:
                isort = np.argsort(np.array([r.index for r in result]))
                result = [result[i] for i in isort]
            return [r.index for r in result] if indices else result
        else:
            return result.index if indices else result

class cell(object):
    """Mesh cell. On creation, the layer and column defining the cell (and
    optionally the cell index) are specified."""

    def __init__(self, lay, col, index = None):
        self.layer = lay #: Cell layer object.
        self.column = col #: Cell column object.
        self.index = index #: Index of the cell in the mesh.

    def __repr__(self):
        return str(self.index)

    def _get_volume(self):
        return self.layer.thickness * self.column.area
    #: Volume of cell.
    volume = property(_get_volume)

    def _get_centroid(self):
        return np.concatenate([self.column.centroid,
                               np.array([self.layer.centre])])
    #: Centroid of cell.
    centroid = property(_get_centroid)
    #: Centroid of cell.
    centre = property(_get_centroid)

    def _get_surface(self):
        return self == self.column.cell[0]
    #: *True* if the cell is at the surface of the mesh, *False* otherwise.
    surface = property(_get_surface)

    def _get_num_nodes(self):
        return 2 * self.column.num_nodes
    #: Number of nodes in the cell (at both top and bottom of layer).
    num_nodes = property(_get_num_nodes)

    def _get_column_layer_cell(self, layer):
        """Returns cell in same column as current cell, in specified layer."""
        c = None
        if layer:
            if self.column.index in layer.column_cell:
                c = layer.column_cell[self.column.index]
        return c

    def _get_above(self):
        return self._get_column_layer_cell(self.layer.above)
    #: Cell above the current cell, or None if there is no cell above it.
    above = property(_get_above)

    def _get_below(self):
        return self._get_column_layer_cell(self.layer.below)
    #: Cell below the current cell, or None if there is no cell below it.
    below = property(_get_below)

    def _get_neighbour(self):
        nbrs = set()
        for col in self.column.neighbour:
            if col.index in self.layer.column_cell:
                nbrs.add(self.layer.column_cell[col.index])
        for c in [self.above, self.below]:
            if c: nbrs.add(c)
        return nbrs
    #: Set of neighbouring cells in the mesh, i.e. those that share a
    #: common face.
    neighbour = property(_get_neighbour)

    def _get_num_neighbours(self):
        return len(self.neighbour)
    #: Number of neighbouring cells in the mesh, i.e. those that share
    #: a common face.
    num_neighbours = property(_get_num_neighbours)

    def _find_layer(self, z):
        """Returns cell layer if it contains the specified elevation *z*, or
        *None* otherwise."""
        return self.layer._find_layer(z)

    def _find_column(self, pos):
        """Returns cell column if it contains the 2-D point pos (tuple, list
        or array of length 2), otherwise *None*."""
        return self.column._find_column(pos)

    def _find_columns(self, polygon):
        """Returns cell column if its centroid is inside the polygon (a tuple,
        list or array of 2-D points, each a tuple, list or array of
        length 2)."""
        return self.column._find_columns(polygon)

    def _find_cell(self, pos):
        """Returns self if the 3-D point (tuple, list or array of length 3)
        *pos* is inside the cell, otherwise *None*."""
        if self.layer._find_layer(pos[2]):
            if self.column._find_column(pos[:2]): return self
            else: return None
        else: return None

    def find(self, match, indices = False):
        """Returns cell, column or layer satisfying the criterion *match*.

        The *match* parameter can be:

        * a **function** taking a cell and returning a Boolean: the cell is
          returned if it matches, otherwise *None*
        * a **scalar**: *match* is interpreted as an **elevation**
          and the cell layer is returned if the elevation is inside it
        * a **2-D point** (tuple, list or array of length 2): *match*
          is interpreted as a **horizontal position**, and the cell
          column is returned if the position is inside it
        * a **polygon** (tuple, list or array of 2-D points): the cell
          column is returned if the cell column centroid is inside the
          polygon
        * a **3-D point** (tuple, list or array of length 3): the cell
          is returned if the point is inside it

        If indices is *True*, the cell, column or layer index
        is returned rather than the cell, column or layer itself.

        In each case, *None* is returned if there is no match.
        """

        if callable(match):
            result = self if match(self) else None
        elif isinstance(match, (float, int)):
            result = self.layer._find_layer(match)
        elif isinstance(match, (tuple, list, np.ndarray)):
            pos = np.array(match)
            ndim = pos.ndim
            if ndim == 1:
                if len(pos) == 1:
                    result = self.layer_.find_layer(pos[0])
                elif len(pos) == 2:
                    result = self.column._find_column(pos)
                elif len(pos) == 3:
                    result = self._find_cell(pos)
                else:
                    raise Exception('Length of point to find is not between 1 and 3.')
            elif ndim == 2:
                if pos.shape[1] == 2:
                    result = self.column._find_columns(pos)
                else:
                    raise Exception('Unrecognised match shape.')
        else:
            raise Exception('Unrecognised match type.')

        if result is None: return None
        else: return result.index if indices else result

class mesh(_layered_object):
    """A mesh can be created either by reading it from a file, or via
        other parameters.

        If *filename* is specified, the mesh is read from the given
        HDF5 file.

        Otherwise, a rectangular mesh can be created using the
        *rectangular* parameter. Mesh spacings in the three coordinate
        directions are specified via tuples, lists or arrays of
        spacings. The *rectangular* parameter is itself a tuple or
        list of three of these mesh spacing specifications.

        The surface elevations can be specified using the *surface*
        parameter. This can be either a dictionary of pairs of column
        indices and corresponding surface elevations, or a tuple, list
        or array of surface elevations for all columns. If *None* is
        specified (the default) then all column surfaces will be set
        to the top of the uppermost layer.

        By default, mesh cells are ordered first by cell type (number
        of nodes, in decreasing order), then layer and finally by
        column within each layer, from the top to bottom of the
        mesh. The sorting of cell types can be reversed or disabled by
        setting the *cell_type_sort* parameter: a value of 1 sorts
        cells in increasing type order, and a value of zero disables
        cell type sorting.

    """

    def __init__(self, filename = None, **kwargs):

        self.node = [] #: List of node objects in the mesh.
        self.column = [] #: List of column objects in the mesh.
        self.layer = [] #: List of layer objects in the mesh.
        self.cell = [] #: List of cell objects in the mesh.

        if filename is not None: self.read(filename)
        else:
            #: Integer controlling sorting of cells by type. A value of -1 (the default)
            #: gives cells sorted by decreasing type (number of nodes). A value of 1
            #: gives cells sorted by increasing type, while a value of zero disables
            #: cell type sorting.
            self.cell_type_sort = kwargs.get('cell_type_sort',
                                             default_cell_type_sort)
            rectangular = kwargs.get('rectangular', None)
            if rectangular is not None:
                thicknesses = rectangular[2]
                elevations = np.hstack((np.zeros(1),
                                       -np.cumsum(np.array(thicknesses))))
                self.set_layers(elevations)
                spacings = rectangular[:2]
                self.set_rectangular_columns(spacings)
            self.surface = kwargs.get('surface', None)

    def __repr__(self):
        return '%d columns, %d layers, %d cells' % \
            (self.num_columns, self.num_layers, self.num_cells)

    def _get_num_nodes(self):
        return len(self.node)
    #: Number of 2-D nodes in the mesh.
    num_nodes = property(_get_num_nodes)

    def _get_num_columns(self):
        return len(self.column)
    #: Number of columns in the mesh.
    num_columns = property(_get_num_columns)

    def _get_num_layers(self):
        return len(self.layer)
    #: Number of layers in the mesh.
    num_layers = property(_get_num_layers)

    def _get_area(self):
        return sum([col.area for col in self.column])
    #: Horizontal area of the mesh.
    area = property(_get_area)

    def _get_volume(self):
        return sum([col.volume for col in self.column])
    #: Total volume of the mesh.
    volume = property(_get_volume)

    def _get_centre(self):
        if self.num_columns > 0:
            c = np.zeros(2)
            a = 0.
            for col in self.column:
                c += col.centre * col.area
                a += col.area
            return c / a
        else: return None
    #: Horizontal centre of the mesh (an array of length 2),
    #: approximated by an area-weighted average of column centres.
    centre = property(_get_centre)

    def _get_bounds(self):
        from layermesh.geometry import bounds_of_points
        return bounds_of_points([node.pos for node in self.node])
    #: Horizontal bounding box for the mesh.
    bounds = property(_get_bounds)

    def _get_boundary_nodes(self):
        bdy = set()
        for col in self.column:
            num_nodes = col.num_nodes
            for i, nbr in enumerate(col.side_neighbour):
                if nbr is None:
                    i1 = (i + 1) % num_nodes
                    for index in [i, i1]:
                        bdy.add(col.node[index])
        return bdy
    #: Set of nodes on the boundary of the mesh.
    boundary_nodes = property(_get_boundary_nodes)

    def add_node(self, n):
        """Adds a horizontal node to the mesh."""
        self.node.append(n)

    def add_layer(self, lay):
        """Adds a layer to the mesh."""
        self.layer.append(lay)
        if self.num_layers > 1:
            self.layer[-1].above = self.layer[-2]
            self.layer[-2].below = self.layer[-1]

    def add_column(self, col):
        """Adds a column to the mesh."""
        self.column.append(col)
        for n in col.node:
            n.column.add(col)

    def delete_column(self, col):
        """Deletes the specified column object from the mesh."""
        for nbr in col.neighbour:
            nbr.neighbour.remove(col)
        for n in col.node:
            n.column.remove(col)
        self.column.remove(col)

    def identify_column_neighbours(self):
        """Identifies neighbours for each column."""
        for col in self.column:
            col.neighbour = set()
            for n in col.node:
                for nbr in n.column:
                    if len(set(col.node) & set(nbr.node)) == 2:
                        col.neighbour.add(nbr)

    def set_layer_columns(self, lay):
        """Populates the list of columns for a given layer."""
        lay.column = [col for col in self.column
                      if self.column_in_layer(col, lay)]

    def setup(self, indices = False):
        """Sets up internal mesh variables, including node, column and layer
        indices if needed."""

        if indices:
            def set_indices(v):
                for i, item in enumerate(v):
                    item.index = i
            set_indices(self.node)
            set_indices(self.column)
            set_indices(self.layer)

        self.identify_column_neighbours()

        for lay in self.layer:
            self.set_layer_columns(lay)

        self.setup_cells()

    def setup_cells(self):
        """Sets up cell properties of mesh, layers and columns."""
        self.cell = []
        for col in self.column: col.cell = []
        if self.cell_type_sort:
            self._setup_cells_type_sorted()
        else:
            self._setup_cells_type_unsorted()

    def _setup_cells_type_sorted(self):
        """Sets up cells, sorted by cell type."""
        cells = {}
        for lay in self.layer:
            lay.cell = []
            lay.column_cell = {}
            for col in lay.column:
                c = cell(lay, col)
                cell_type = c.num_nodes
                if cell_type not in cells: cells[cell_type] = []
                cells[cell_type].append(c)
                lay.cell.append(c)
                lay.column_cell[col.index] = c
                col.cell.append(c)

        cell_types = cells.keys()
        if self.cell_type_sort > 0:
            cell_types = sorted(cell_types)
        elif self.cell_type_sort < 0:
            cell_types = sorted(cell_types, reverse = True)
        else:
            raise Exception('Unrecognised cell type sort: %s' % str(self.cell_type_sort))

        index = 0
        for cell_type in cell_types:
            if cell_type in cells:
                for c in cells[cell_type]:
                    c.index = index
                    self.cell.append(c)
                    index += 1

    def _setup_cells_type_unsorted(self):
        """Sets up cells, not sorted by cell type."""
        index = 0
        for lay in self.layer:
            lay.cell = []
            lay.column_cell = {}
            for col in lay.column:
                c = cell(lay, col, index)
                self.cell.append(c)
                lay.cell.append(c)
                lay.column_cell[col.index] = c
                col.cell.append(c)
                index += 1

    def _get_num_cells(self):
        return len(self.cell)
    #: Number of 3-D cells in the mesh.
    num_cells = property(_get_num_cells)

    def set_rectangular_columns(self, spacings):
        """Sets rectangular mesh columns according to specified lists of horizontal
        mesh spacings."""

        self.node = []
        self.column = []
        verts = [np.concatenate([np.zeros(1), np.cumsum(np.array(s))])
                 for s in spacings]
        nv0 = len(verts[0])
        ns = [len(s) for s in spacings]
        index = 0
        for y in verts[1]:
            for x in verts[0]:
                n = node(pos = [x, y], index = index)
                self.add_node(n); index += 1
        index = 0
        for j in range(ns[1]):
            for i in range(ns[0]):
                column_node_indices = [
                    j * nv0 + i,
                    (j + 1) * nv0 + i,
                    (j + 1) * nv0 + i + 1,
                    j * nv0 + i + 1]
                column_nodes = [self.node[ind]
                                for ind in column_node_indices]
                col = column(node = column_nodes, index = index)
                self.add_column(col)
                index += 1

    def set_layers(self, elevations):
        """Sets mesh layers according to specified vertical layer boundary
        elevations, from the top down. """

        self.layer = []
        index = 0
        bottom = elevations[0]
        for z in elevations[1:]:
            top = bottom
            bottom = z
            lay = layer(bottom, top, index)
            self.add_layer(lay)
            index += 1

    def column_in_layer(self, col, lay):
        """Returns *True* if column is in the specified layer, or
        *False* otherwise."""
        return col.num_layers >= self.num_layers - lay.index

    def set_surface(self, surface):
        """Sets column layers from surface dictionary (keyed by
        column indices) or list/array of values for all columns."""

        if surface is None:
           for col in self.column: col.set_surface(self.layer)
        elif isinstance(surface, dict):
            for col in self.column:
                if col.index in surface:
                    col.set_surface(self.layer, surface[col.index])
                else:
                    col.set_surface(self.layer)
        elif isinstance(surface, (tuple, list, np.ndarray)):
             if len(surface) == self.num_columns:
                 for col, s in zip(self.column, surface):
                     col.set_surface(self.layer, s)
             else:
                 raise Exception('Surface is the wrong size.')
        else:
            raise Exception('Unrecognized surface parameter type.')
        self.setup()

    def _get_surface(self):
        return np.array([col.surface for col in self.column])
    #: Array of column surface elevations.
    surface = property(_get_surface, set_surface)

    def set_column_layers(self, num_layers):
        """Sets column layers from dictionary (keyed by column indices) or
        list/array of layer counts for each column."""
        if num_layers is None:
           for col in self.column: col.set_surface(self.layer)
        elif isinstance(num_layers, dict):
            for col in self.column:
                if col.index in num_layers:
                    col.set_layers(self.layer, num_layers[col.index])
                else:
                    col.set_layers(self.layer)
        elif isinstance(num_layers, (list, np.ndarray)):
             if len(num_layers) == self.num_columns:
                 for col, n in zip(self.column, num_layers):
                     col.set_layers(self.layer, n)
             else:
                 raise Exception('num_layers is the wrong size.')
        else:
            raise Exception('Unrecognized num_layers parameter type.')
        self.setup()

    def write(self, filename):
        """Writes mesh to HDF5 file."""
        import h5py
        with h5py.File(filename, 'w') as f:
            cell_group = f.create_group('cell')
            cell_group.create_dataset('type_sort', data = self.cell_type_sort)
            if self.layer:
                layer_elev = [self.layer[0].top] + \
                         [lay.bottom for lay in self.layer]
                lay_group = f.create_group('layer')
                dset = lay_group.create_dataset('elevation', data = layer_elev)
                dset.attrs['description'] = 'Layer boundary elevations, ' + \
                                            'from top to bottom'
            if self.node:
                pos = np.array([n.pos for n in self.node])
                node_group = f.create_group('node')
                dset = node_group.create_dataset('position', data = pos)
                dset.attrs['description'] = 'Position of each node'
                if self.column:
                    max_col_nodes = max([col.num_nodes for col in self.column])
                    col_node_indices = np.full((self.num_columns, max_col_nodes), -1,
                                               dtype = int)
                    for i, col in enumerate(self.column):
                        col_node_indices[i, 0: col.num_nodes] = [n.index for n in col.node]
                    col_group = f.create_group('column')
                    dset = col_group.create_dataset('node', data = col_node_indices)
                    dset.attrs['description'] = 'Indices of nodes in each column'
                    num_layers = np.array([col.num_layers for col in self.column])
                    dset = col_group.create_dataset('num_layers', data = num_layers)
                    dset.attrs['description'] = 'Number of layers in each column'

    def read(self, filename):
        """Reads mesh from HDF5 file."""
        import h5py
        num_layers = None
        with h5py.File(filename, 'r') as f:
            if 'cell' in f:
                cell_group = f['cell']
                if 'type_sort' in cell_group:
                    self.cell_type_sort = int(np.array(cell_group['type_sort']))
            if 'layer' in f:
                lay_group = f['layer']
                self.set_layers(np.array(lay_group['elevation']))
            if 'node' in f:
                node_group = f['node']
                if 'position' in node_group:
                    for index, p in enumerate(np.array(node_group['position'])):
                        n = node(pos = p, index = index)
                        self.add_node(n)
                    if 'column' in f:
                        col_group = f['column']
                        if 'node' in col_group:
                            for index, col_node_indices in \
                                enumerate(np.array(col_group['node'])):
                                col_nodes = [self.node[i]
                                             for i in col_node_indices if i >= 0]
                                col = column(node = col_nodes, index = index)
                                self.add_column(col)
                            if 'num_layers' in col_group:
                                num_layers = np.array(col_group['num_layers'])
        self.set_column_layers(num_layers)

    def _get_meshio_points_cells(self):

        points = []
        cells = {'wedge': [], 'hexahedron': []}
        cell_type = {6: 'wedge', 8: 'hexahedron'}
        node_index = {}
        point_index = 0

        for c in self.cell:
            elt = []
            for iz, z in enumerate([c.layer.top, c.layer.bottom]):
                ilayer = c.layer.index + iz
                for n in c.column.node:
                    k = (ilayer, n.index)
                    if k not in node_index: # create point:
                        pos = np.concatenate((n.pos, np.array([z])))
                        points.append(pos)
                        node_index[k] = point_index
                        point_index += 1
                    elt.append(node_index[k])
            cells[cell_type[len(elt)]].append(elt)

        points = np.array(points)
        cells = dict([(k, np.array(v)) for k, v in cells.items() if v])
        return points, cells

    #: Lists of 3-D points and cells suitable for mesh
    #: input/output using meshio library.
    meshio_points_cells = property(_get_meshio_points_cells)

    def export(self, filename, fmt = None):
        """Exports 3-D mesh using meshio, to file with the specified name. If
        the format is not specified via the fmt parameter, it is determined
        from the filename extension."""
        import meshio
        points, cells = self.meshio_points_cells
        meshio.write_points_cells(filename, points, cells, file_format = fmt)

    def _get_surface_cells(self):
        return [col.cell[0] for col in self.column]
    #: List of cells at the surface of the mesh.
    surface_cells = property(_get_surface_cells)

    def column_faces(self, columns = None):
        """Returns a list of the column faces between the specified columns. A
        list of the columns may be optionally specified, otherwise all
        columns will be included.
        """
        if columns is None: columns = self.column
        col_dict = {col.index: col for col in columns}
        face_keys = set()
        for col in columns:
            for nbr in col.neighbour:
                if nbr.index in col_dict:
                    face_keys.add(frozenset([col.index, nbr.index]))
        faces = []
        for key in face_keys:
            cols = [self.column[col] for col in key]
            face = column_face(cols)
            faces.append(face)
        return faces

    def nodes_in_columns(self, columns):
        """Returns a set of nodes in the specified columns."""
        nodes = set()
        for col in columns:
            nodes = nodes | set(col.node)
        return nodes

    def type_columns(self, num_nodes):
        """Returns a list of mesh columns of a specified type, i.e. number of
        nodes."""
        return [col for col in self.column if col.num_nodes == num_nodes]

    def translate(self, shift):
        """Translates the mesh by the specified 3-D shift vector (tuple, list
        or array of length 3)."""
        shift = np.array(shift)
        for node in self.node: node.pos += shift[:2]
        for col in self.column: col.translate(shift[:2])
        for layer in self.layer: layer.translate(shift)

    def rotate(self, angle, centre = None):
        """Rotates the mesh horizontally by the specified angle (degrees
        clockwise). If no centre is specified, the mesh is rotated
        about its own centre."""
        from layermesh.geometry import rotation
        if centre is None:
            c = self.centre
        else:
            c = np.array(centre)
        A, b = rotation(angle, c)
        for n in self.node: n.pos = np.dot(A, n.pos) + b
        for col in self.column:
            if col._centroid is not None:
                col._centroid = np.dot(A, col._centroid) + b
        for lay in self.layer:
            lay._quadtree = None

    def _find_column(self, pos):
        """Returns column containing the 2-D point (tuple, list or array of
        length 2) *pos*."""
        if self.num_layers == 0: return None
        else: return self.layer[-1]._find_column(pos)

    def _find_columns(self, polygon):
        """Returns a list of columns with centroids inside the polygon
        (tuple, list or array of 2-D points, each a tuple, list or
        array of length 2).
        """
        if self.num_layers == 0: return []
        else: return self.layer[-1]._find_columns(polygon)

    def _find_cell(self, pos):
        """Returns the cell containing the 3-D point *pos* (list, tuple or
        array of length 3), or *None* if *pos* is outside the mesh.
        """
        lay = self._find_layer(pos[2])
        if lay is None: return None
        else: return lay._find_cell(pos)

    def find(self, match, indices = False, sort = False):
        """Returns cells, columns or layers satisfying the criterion *match*.

        The *match* parameter can be:

        * a **function** taking a cell and returning a Boolean: a list
          of matching cells is returned
        * a **scalar**: *match* is interpreted as an **elevation**,
          and the layer containing it is returned
        * a **2-D point** (tuple, list or array of length 2): *match*
          is interpreted as a **horizontal position**, and the mesh column
          containing it is returned
        * a **polygon** (tuple, list or array of 2-D points): a list of
          mesh columns inside the polygon are returned
        * a **3-D point** (tuple, list or array of length 3): *match*
          is interpreted as a **3-D position**, and the mesh cell
          containing it is returned

        If indices is *True*, the cell, column or layer indices
        are returned rather than the cells, columns or layers
        themselves.

        If *sort* is *True*, then lists of results are sorted by index.

        If no match is found, then *None* is returned, except when the
        expected result is a list, in which case an empty list is
        returned.
        """

        if callable(match):
            result = [c for c in self.cell if match(c)]
        elif isinstance(match, (float, int)):
            result = self._find_layer(match)
        elif isinstance(match, (tuple, list, np.ndarray)):
            pos = np.array(match)
            if pos.ndim == 1:
                if len(pos) == 1:
                    result = self._find_layer(pos[0])
                elif len(pos) == 2:
                    result = self._find_column(pos)
                elif len(pos) == 3:
                    result = self._find_cell(pos)
                else:
                    raise Exception('Length of point to find is not between 1 and 3.')
            elif pos.ndim == 2:
                if pos.shape[1] == 2:
                    result = self._find_columns(pos)
                else:
                    raise Exception('Unrecognised match shape.')
            else:
                raise Exception('Unrecognised match shape.')
        else:
            raise Exception('Unrecognised match type.')

        if result is None: return None
        elif isinstance(result, list):
            if sort:
                isort = np.argsort(np.array([r.index for r in result]))
                result = [result[i] for i in isort]
            return [r.index for r in result] if indices else result
        else:
            return result.index if indices else result

    def column_track(self, line):
        """Returns a list of tuples of (column, entry_point, exit_point)
        representing the horizontal track traversed by the specified
        line through the grid.  Line is a tuple of two 2D arrays.

        The resulting list is ordered by distance from the start of the
        line. Adapted from the PyTOUGH mulgrid column_track()
        method."""

        def furthest_intersection(poly, line):
            """Returns furthest intersection point between line and poly."""
            from layermesh.geometry import line_polygon_intersections
            pts, inds = line_polygon_intersections(poly, line,
                                                   bound_line = (True, False),
                                                   indices = True)
            if pts:
                d = np.array([np.linalg.norm(intpt - line[0]) for intpt in pts])
                i = np.argmax(d)
                return pts[i], inds[i]
            else: return None, None

        def find_track_start(line):
            """Finds starting point for track- an arbitrary point on the line that
            is inside the grid.  If the start point of the line is
            inside the grid, that is used; otherwise, a recursive
            bisection technique is used to find a point."""

            col, start_type = None, None
            for endpt, name in zip(line, ['start', 'end']):
                pos, col, start_type = endpt, self.find(endpt), name
                if col: break
            if not col: # line ends are both outside the grid:
                start_type = 'mid'
                max_levels = 7

                def find_start(line, level = 0):
                    midpt = 0.5 * (line[0] + line[1])
                    col = self.find(midpt)
                    if col: return midpt, col
                    else:
                        if level <= max_levels:
                            line0, line1 = [line[0], midpt], [midpt, line[1]]
                            pos, col = find_start(line0, level + 1)
                            if col: return pos, col
                            else:
                                pos, col = find_start(line1, level + 1)
                                if col: return pos, col
                                else: return None, None
                        else: return None, None

                pos, col = find_start(line)
            return pos, col, start_type

        def next_corner_column(col, pos, more, cols):
            """If the line has hit a node, determine a new column containing that
            node, not already visited."""

            node_tol = 1.e-12
            nextcol = None
            nearnodes = [n for n in col.node if np.linalg.norm(n.pos - pos) < node_tol]
            if nearnodes: # hit a node
                nearnode = nearnodes[0]
                nearcols = nearnode.column - cols
                if nearcols: nextcol = nearcols.pop()
                else: more = False
            return nextcol, more

        def next_neighbour_column(col, more, cols):
            """Determine a new neighbour column not already visited."""

            nbrs = col.neighbour - cols
            if nbrs: return nbrs.pop(), more
            else: return None, False

        def find_track_segment(linesegment, pos, col):
            """Finds track segment starting from the specified position and
            column."""
            track = []
            cols, more, inpos = set(), True, pos
            colnbr, nextcol = col.side_neighbour, None
            lined = np.linalg.norm(linesegment[1] - linesegment[0])
            while more:
                cols.add(col)
                outpos, ind = furthest_intersection(col.polygon, linesegment)
                if outpos is not None:
                    d = np.linalg.norm(outpos - linesegment[0])
                    if d >= lined: # gone past end of line
                        outpos = linesegment[1]
                        more = False
                    if np.linalg.norm(outpos - inpos) > 0.:
                        track.append(tuple([col, inpos, outpos]))
                    if more: # find next column
                        inpos = outpos
                        nextcol = colnbr[ind]
                        if nextcol:
                            if nextcol in cols:
                                nextcol, more = next_corner_column(col, outpos, more, cols)
                                if nextcol is None:
                                    nextcol, more = next_neighbour_column(col, more, cols)
                                    nbr_base_col = col
                        else: nextcol, more = next_corner_column(col, outpos, more, cols)
                else:
                    nextcol, more = next_neighbour_column(nbr_base_col, more, cols)
                col = nextcol
                if col: colnbr = col.side_neighbour
                else: more = False

            return track

        def reverse_track(track):
            return [tuple([tk[0], tk[2], tk[1]]) for tk in track][::-1]

        line = [np.array(p) for p in line]
        pos, col, start_type = find_track_start(line)
        if pos is not None and col:
            if start_type == 'start':
                track = find_track_segment(line, pos, col)
            elif start_type == 'end':
                track = find_track_segment(line[::-1], pos, col)
                track = reverse_track(track)
            else:
                track1 = find_track_segment([pos, line[0]], pos, col)
                track2 = find_track_segment([pos, line[1]], pos, col)
                # remove arbitrary starting point from middle of track, and join:
                midtk = tuple([track1[0][0], track1[0][2], track2[0][2]])
                track = reverse_track(track1)[:-1] + [midtk] + track2[1:]
            return track
        else: return []

    def layer_plot(self, lay = -1, **kwargs):
        """Creates a 2-D Matplotlib plot of the mesh at a specified layer. The
        *lay* parameter can be either a layer object or a layer index.

        Other optional parameters:

        * *aspect*: the aspect ratio of the axes (default *'equal'*).
        * *axes*: a Matplotlib axes object on which to draw the plot. If not
          specified, then a new axes object will be created internally.
        * *colourmap*: a Matplotlib colourmap object for shading the plot
          according to the *value* array (default *None*).
        * *elevation*: used to specify an elevation instead of a layer.
        * *label*: a string (or *None*, the default) specifying what labels\
          are to be drawn at the centre of each column. Possible values are
          *'column'* (to label with column indices), *'cell'* (to label cell
          indices) or *'value'* (to label with the *value* array).
        * *label_format*: format string for the labels (default *'%g'*).
        * *label_colour*: the colour of the labels (default *'black'*).
        * *linecolour*: the colour of the mesh grid (default *'black'*).
        * *linewidth*: the line width of the mesh (default *0.2*).
        * *value*: a tuple, list or array of values to plot over the mesh,
          of length equal to the number of cells in the mesh.
        * *xlabel*: label string for the plot *x*-axis (default *'x'*).
        * *ylabel*: label string for the plot *y*-axis (default *'y'*).

        """

        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if 'axes' in kwargs: ax = kwargs['axes']
        else: fig, ax = plt.subplots()

        if 'elevation' in kwargs:
            z = kwargs['elevation']
            lay = self.find(z)
            if lay is None:
                raise Exception('Elevation not found in layer_plot()')
        else:
            if isinstance(lay, layer):
                if lay not in self.layer:
                    raise Exception('Unknown layer in layer_plot()')
            elif isinstance(lay, int):
                try:
                    lay = self.layer[lay]
                except:
                    raise Exception('Unknown layer in layer_plot()')

        labels = kwargs.get('label', None)
        label_fmt = kwargs.get('label_format', '%g')
        label_colour = kwargs.get('label_colour', 'black')
        verts = []
        for c in lay.cell:
            col = c.column
            poslist = [tuple([p for p in n.pos])
                                for n in col.node]
            verts.append(tuple(poslist))
            if labels == 'column':
                col_label = label_fmt % col.index
            elif labels == 'cell':
                col_label = label_fmt % c.index
            else: col_label = None
            if col_label:
                ax.text(col.centre[0], col.centre[1], col_label,
                        clip_on = True, horizontalalignment = 'center',
                        color = label_colour)

        linewidth = kwargs.get('linewidth', 0.2)
        linecolour = kwargs.get('linecolour', 'black')
        colourmap = kwargs.get('colourmap', None)
        polys = collections.PolyCollection(verts,
                                           linewidth = linewidth,
                                           facecolors = [],
                                           edgecolors = linecolour,
                                           cmap = colourmap)
        ax.add_collection(polys)

        if 'value' in kwargs:
            vals = kwargs['value']
            if len(vals) >= self.num_cells:
                vals = np.array(kwargs['value'])
                indices = [c.index for c in lay.cell]
                layer_vals = vals[indices]
                polys.set_array(layer_vals)
                self._plot_colourbar(ax, polys, kwargs)
                if labels == 'value':
                    for c in lay.cell:
                        col = c.column
                        col_label = label_fmt % vals[c.index]
                        ax.text(col.centre[0], col.centre[1], col_label,
                                clip_on = True, horizontalalignment = 'center',
                                color = label_colour)
            else:
                raise Exception('Not enough values for mesh in layer_plot()')

        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))

        ax.set_aspect(kwargs.get('aspect', 'equal'))
        ax.autoscale_view()

        if 'axes' not in kwargs: plt.show()

    def _plot_colourbar(self, ax, polys, kwargs):
        """Adds colour bar to Matplotlib plot on specified axes."""

        import matplotlib.pyplot as plt

        cbar = plt.colorbar(polys, ax = ax)
        if 'value_label' in kwargs:
            value_label = kwargs.get('value_label')
            if 'value_unit' in kwargs:
                unit = kwargs.get('value_unit')
                value_label += ' (' + unit + ')'
            cbar.set_label(value_label)

    def slice_plot(self, line = 'x', **kwargs):
        """Creates a 2-D Matplotlib plot of the mesh through a specified
        vertical slice. The horizontal *line* defining the slice can be either:

        * *'x'*: to plot through the mesh centre along the *x*-axis
        * *'y'*: to plot through the mesh centre along the *y*-axis
        * a number representing an angle (in degrees), clockwise from the *y*-axis,
          to plot through the centre of the mesh at that angle
        * a tuple, list or array of two end-points for the line, each point being
          itself a tuple, list or array of length 2

        Other optional parameters:

        * *aspect*: the aspect ratio of the axes (default *'auto'*).
        * *axes*: a Matplotlib axes object on which to draw the plot. If not
          specified, then a new axes object will be created internally.
        * *colourmap*: a Matplotlib colourmap object for shading the plot
          according to the *value* array (default *None*).
        * *label*: a string (or *None*, the default) specifying what labels\
          are to be drawn at the centre of each cell. Possible values are
          *'cell'* (to label cell indices) or *'value'* (to label with the
          *value* array).
        * *label_format*: format string for the labels (default *'%g'*).
        * *label_colour*: the colour of the labels (default *'black'*).
        * *linecolour*: the colour of the mesh grid (default *'black'*).
        * *linewidth*: the line width of the mesh (default *0.2*).
        * *value*: a tuple, list or array of values to plot over the mesh,
          of length equal to the number of cells in the mesh.
        * *xlabel*: label string for the plot *x*-axis (default *'x'*).
        * *ylabel*: label string for the plot *y*-axis (default *'z'*).

        """

        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if 'axes' in kwargs: ax = kwargs['axes']
        else: fig, ax = plt.subplots()

        if line == 'x':
            bounds = self.bounds
            l = ([bounds[0][0], self.centre[1]],
                 [bounds[1][0], self.centre[1]])
        elif line == 'y':
            bounds = self.bounds
            l = ([self.centre[0], bounds[0][1]],
                 [self.centre[0], bounds[1][1]])
        elif isinstance(line, (float, int)):
            # line through mesh centre at specified angle in degrees:
            bounds = self.bounds
            r = 0.5 * np.linalg.norm(bounds[1] - bounds[0])
            from math import radians, cos, sin
            theta = radians(linespec)
            d = r * np.array([sin(theta), cos(theta)])
            l = [self.centre - d, self.centre + d]
        else: l = line

        l = [np.array(p) for p in l]

        if np.linalg.norm(l[1] - l[0]) > 0.0:

            labels = kwargs.get('label', None)
            label_fmt = kwargs.get('label_format', '%g')
            label_colour = kwargs.get('label_colour', 'black')
            slice_cells = []
            verts = []
            dcol = {}
            track = self.column_track(l)
            for item in track:
                col, points = item[0], item[1:]
                inpoint = points[0]
                if len(points) > 1: outpoint = points[1]
                else: outpoint = inpoint
                if line == 'x':
                    din, dout = inpoint[0], outpoint[0]
                    default_xlabel = 'x'
                elif line == 'y':
                    din, dout = inpoint[1], outpoint[1]
                    default_xlabel = 'y'
                else:
                    din = np.linalg.norm(inpoint - l[0])
                    dout = np.linalg.norm(outpoint - l[0])
                    default_xlabel = 'distance'
                dcol[col.index] = 0.5 * (din + dout)
                for c in col.cell:
                    slice_cells.append(c)
                    verts.append(((din, c.layer.bottom),
                                  (din, c.layer.top),
                                  (dout, c.layer.top),
                                  (dout, c.layer.bottom)))
                    if labels == 'cell':
                        cell_label = label_fmt % c.index
                    else: cell_label = None
                    if cell_label:
                        ax.text(dcol[col.index], c.layer.centre, cell_label,
                                clip_on = True, horizontalalignment = 'center',
                                color = label_colour)

            linewidth = kwargs.get('linewidth', 0.2)
            linecolour = kwargs.get('linecolour', 'black')
            colourmap = kwargs.get('colourmap', None)
            polys = collections.PolyCollection(verts,
                                               linewidth = linewidth,
                                               facecolors = [],
                                               edgecolors = linecolour,
                                               cmap = colourmap)
            ax.add_collection(polys)

            if 'value' in kwargs:
                vals = kwargs['value']
                if len(vals) >= self.num_cells:
                    vals = np.array(kwargs['value'])
                    indices = [c.index for c in slice_cells]
                    slice_vals = vals[indices]
                    polys.set_array(slice_vals)
                    self._plot_colourbar(ax, polys, kwargs)
                    if labels == 'value':
                        for c in slice_cells:
                            col = c.column
                            cell_label = label_fmt % vals[c.index]
                            ax.text(dcol[col.index], c.layer.centre, cell_label,
                                    clip_on = True, horizontalalignment = 'center',
                                    color = label_colour)
                else:
                    raise Exception('Not enough values for mesh in slice_plot()')

            ax.set_xlabel(kwargs.get('xlabel', default_xlabel))
            ax.set_ylabel(kwargs.get('ylabel', 'z'))

            ax.set_aspect(kwargs.get('aspect', 'auto'))
            ax.autoscale_view()

            if 'axes' not in kwargs: plt.show()

        else:
            raise Exception('Line of zero length in slice_plot()')

    def fit_data_to_columns(self, data, columns = None, smoothing = 0.01):
        """Fits scattered data to the columns of the mesh, using smoothed
        piecewise constant least-squares fitting.

        The data should be in the form of a 3-column array with x,y,z
        data in each row. Fitting can be carried out over a subset of
        the mesh columns by specifying a tuple or list of columns.

        Increasing the smoothing parameter will decrease gradients
        between columns, and a non-zero value must be used to obtain a
        solution if any columns contain no data.

        """

        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        if columns is None: columns = self.column
        col_dict = {col.index: i for i, col in enumerate(columns)}

        N = len(columns)
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)

        for d in data:
            col = self.find(d[:2])
            if col:
                if col.index in col_dict:
                    i = col_dict[col.index]
                    A[i,i] += 1
                    b[i] += d[2]

        faces = self.column_faces(columns)
        for f in faces:
            i = [col_dict[col.index] for col in f.column]
            A[i[0], i[0]] += smoothing
            A[i[0], i[1]] -= smoothing
            A[i[1], i[1]] += smoothing
            A[i[1], i[0]] -= smoothing

        A = A.tocsr()
        return spsolve(A, b)

    def fit_surface(self, data, columns = None, smoothing = 0.01):
        """Fits surface elevation data to determine the number of layers in
        each column.

        The *data* should be in the form of a 3-column array with
        x,y,z data in each row. Fitting can be carried out over a
        subset of the mesh columns by specifying a tuple or list of
        columns.

        Increasing the smoothing parameter will decrease gradients
        between columns, and a non-zero value must be used to obtain a
        solution if any columns contain no data.

        """

        if columns is None: columns = self.column

        z = self.fit_data_to_columns(data, columns, smoothing)

        for col, s in zip(columns, z):
            col.set_surface(self.layer, s)
        self.setup()

    def refine(self, columns = None):
        """Refines selected columns in the mesh. If no columns are specified,
        then all columns are refined. The *columns* parameter can be a
        set, tuple or list of column objects.

        Each selected column is divided into four new
        columns. Triangular transition columns are added around the
        edge of the refinement area as needed.

        Note that the selected columns must be either triangular or
        quadrilateral (columns with more than four edges cannot be
        refined).

        Mesh refinement will generally alter the indexing of the mesh
        nodes, columns and cells, even those not within the refinement
        area. Hence, it should not be assumed, for example, that
        columns outside the refinement area will retain their original
        indices after the refinement.

        Based on the mulgrid refine() method in PyTOUGH.

        """

        from copy import copy

        def add_midpoint_node(nodes, midpoint_nodes):
            """Create node at midpoint between two nodes, and add it to the
            dictionary of midpoint nodes, indexed by the original node
            indices."""
            midpos = 0.5 * (nodes[0].pos + nodes[1].pos)
            mid = node(midpos)
            self.add_node(mid)
            ind = frozenset((nodes[0].index, nodes[1].index))
            midpoint_nodes[ind] = mid
            return midpoint_nodes

        def transition_type(num_nodes, sides):
            """Returns transition type- classified by how many
            refined sides, starting side, and range"""
            num_refined = len(sides)
            missing = list(set(range(num_nodes)) - set(sides))
            num_unrefined = len(missing)
            if num_refined == 1:
                return 1, sides[0], 0
            elif num_refined == num_nodes:
                return num_nodes, 0, num_nodes - 1
            elif num_unrefined == 1:
                return num_refined, (missing[0] + 1) % num_nodes, num_nodes - 2
            elif num_nodes == 4 and num_refined == 2:
                diff = sides[1] - sides[0]
                if diff < 3: return num_refined, sides[0], diff
                else: return num_refined, sides[1], 1
            else:
                raise Exception('Unhandled transition in refine().')

        # How to subdivide a column, based on number of nodes, number of
        # refined sides and range of refined sides:
        transition_column = {3: {(1, 0): ((0, (0, 1), 2), ((0, 1), 1, 2)),
                                 (2, 1): ((0, (0, 1), (1, 2), 2), ((0, 1), 1, (1, 2))),
                                 (3, 2): ((0, (0, 1), (2, 0)), ((0, 1), 1, (1, 2)),
                                          ((1, 2), 2, (2, 0)), ((0, 1), (1, 2), (2, 0)))},
                             4: {(1, 0): ((0, (0, 1), 3), ((0, 1), 1, 2),
                                          ((0, 1), 2, 3)),
                                 (2, 1): ((0, (0, 1), 'c'), ((0, 1), 1, 'c'),
                                          (1, (1, 2), 'c'), ((1, 2), 2, 'c'),
                                          (2, 3, 'c'), (0, 'c', 3)),
                                 (2, 2): ((0, (0, 1), (2, 3), 3),
                                          ((0, 1), 1, 2, (2, 3))),
                                 (3, 2): ((0, (0, 1), (2, 3), 3),
                                          ((0, 1), 1, (1, 2)), ((1, 2), 2, (2, 3)),
                                          ((0, 1), (1, 2), (2, 3))),
                                 (4, 3): ((0, (0, 1), 'c', (3, 0)),
                                          ((0, 1), 1, (1, 2), 'c'),
                                          ((1, 2), 2, (2, 3), 'c'),
                                          ((2, 3), 3, (3, 0), 'c'))}}

        if columns is None: columns = self.column
        columns = set(columns)
        faces = self.column_faces(columns)
        bdy = self.boundary_nodes

        halo = set()
        for col in columns:
            halo_nbrs = col.neighbour - columns
            halo = halo | halo_nbrs

        for col in halo:
            for nbr in col.neighbour & columns:
                faces.append(column_face([col, nbr]))

        midpoint_nodes = {}
        for f in faces:
            midpoint_nodes = add_midpoint_node(f.node, midpoint_nodes)
        for col in columns:
            num_nodes = col.num_nodes
            for i, corner in enumerate(col.node):
                next_corner = col.node[(i + 1) % num_nodes]
                if corner in bdy and next_corner in bdy:
                    nodes = [corner, next_corner]
                    midpoint_nodes = add_midpoint_node(nodes, midpoint_nodes)

        for col in columns | halo:

            num_nodes = col.num_nodes
            refined_sides = []
            for i, corner in enumerate(col.node):
                next_corner = col.node[(i + 1) % num_nodes]
                ind = frozenset((corner.index, next_corner.index))
                if ind in midpoint_nodes: refined_sides.append(i)

            num_refined, istart, irange = transition_type(num_nodes,
                                                          refined_sides)
            sub_cols = transition_column[num_nodes][num_refined, irange]
            if any(['c' in local_nodes for local_nodes in sub_cols]):
                centre_node = node(col.centre)
                self.add_node(centre_node)

            for local_nodes in sub_cols:
                nodes = []
                for vertex in local_nodes:
                    if isinstance(vertex, int):
                        n = col.node[(istart + vertex) % num_nodes]
                    elif vertex == 'c':
                        n = centre_node
                    else:
                        ind = frozenset([col.node[(istart + i) % num_nodes].index
                                         for i in vertex])
                        n = midpoint_nodes[ind]
                    nodes.append(n)
                sub_col = column(nodes)
                sub_col.layer = copy(col.layer)
                self.add_column(sub_col)

        while columns:
            col = columns.pop()
            self.delete_column(col)
        while halo:
            col = halo.pop()
            self.delete_column(col)

        self.setup(indices = True)

    def optimize(self, nodes = None, columns = None,
                 weight = {'orthogonal': 1}):
        """Adjusts horizontal positions of specified nodes to optimize the
        mesh. If no nodes are specified, all node positions are
        optimized. If columns are specified, the positions of nodes in
        those columns are optimized.

        Three types of optimization are offered, with their relative
        importance in the optimization specified via the *weight*
        dictionary parameter. This can contain up to three keys:

        * 'orthogonal': the orthogonality of the mesh faces
        * 'skewness': the skewness of the columns
        * 'aspect': the aspect ratio of the columns

        Omitting any of these keys from the *weight* parameter will
        give them zero weight. Weights need not sum to 1: only their
        relative magnitudes matter.

        The optimization is carried out by using the ``leastsq()``
        function from the ``scipy.optimize`` module to minimize an
        objective function formed from a weighted combination of the
        mesh quality measures above.

        """

        from scipy.optimize import leastsq

        if nodes is None:
            if columns is None:
                nodes = self.node
            else:
                nodes = self.nodes_in_columns(columns)
        nodes = list(nodes)
        num_nodes = len(nodes)

        cols = set()
        for node in nodes:
            cols = cols | node.column

        faces = self.column_faces(cols)

        halo = set()
        for col in cols:
            halo_nbrs = col.neighbour - cols
            halo = halo | halo_nbrs

        for col in halo:
            for nbr in col.neighbour & cols:
                faces.append(column_face([col, nbr]))

        def update(x):
            positions = x.reshape((num_nodes, 2))
            for n, pos in zip(nodes, positions): n.pos = pos
            for col in cols:
                col._centroid = None

        def f(x):
            update(x)
            result = []
            if 'orthogonal' in weight:
                angle_cosines = [face.angle_cosine for face in faces]
                result += list(weight['orthogonal'] * np.array(angle_cosines))
            if 'skewness' in weight:
                skewness = [col.angle_ratio - 1 for col in cols]
                result += list(weight['skewness'] * np.array(skewness))
            if 'aspect' in weight:
                aspect = [col.face_length_ratio - 1 for col in cols]
                result += list(weight['aspect'] * np.array(aspect))
            return np.array(result)

        x0 = np.array([n.pos for n in nodes]).reshape(2 * num_nodes)
        x1, success = leastsq(f, x0)
        if success > 4:
            raise Exception('No convergence in optimize().')
        update(x1)
        for col in cols: col._area = None
