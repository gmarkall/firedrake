# Utility functions to derive global and local numbering from DMPlex
from petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc

from libc.string cimport memset
from libc.stdlib cimport qsort

np.import_array()

include "dmplex.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def facet_numbering(PETSc.DM plex, kind,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] facets,
                    PETSc.Section cell_numbering,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closures):
    """Compute the parent cell(s) and the local facet number within
    each parent cell for each given facet.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg kind: String indicating the facet kind (interior or exterior)
    :arg facets: Array of input facets
    :arg cell_numbering: Section describing the global cell numbering
    :arg cell_closures: 2D array of ordered cell closures
    """
    cdef:
        PetscInt f, fStart, fEnd, fi, cell
        PetscInt nfacets, nclosure, ncells, cells_per_facet
        PetscInt *cells = NULL
        np.ndarray[np.int32_t, ndim=2, mode="c"] facet_cells
        np.ndarray[np.int32_t, ndim=2, mode="c"] facet_local_num

    fStart, fEnd = plex.getHeightStratum(1)
    nfacets = facets.shape[0]
    nclosure = cell_closures.shape[1]

    assert(kind in ["interior", "exterior"])
    if kind == "interior":
        cells_per_facet = 2
    else:
        cells_per_facet = 1
    facet_local_num = np.empty((nfacets, cells_per_facet), dtype=np.int32)
    facet_cells = np.empty((nfacets, cells_per_facet), dtype=np.int32)

    # First determine the parent cell(s) for each facet
    for f in range(nfacets):
        CHKERR(DMPlexGetSupport(plex.dm, facets[f], &cells))
        CHKERR(DMPlexGetSupportSize(plex.dm, facets[f], &ncells))
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, cells[0], &cell))
        facet_cells[f,0] = cell
        if cells_per_facet > 1:
            if ncells > 1:
                CHKERR(PetscSectionGetOffset(cell_numbering.sec,
                                             cells[1], &cell))
                facet_cells[f,1] = cell
            else:
                facet_cells[f,1] = -1

    # Run through the sorted closure to get the
    # local facet number within each parent cell
    for f in range(nfacets):
        # First cell
        cell = facet_cells[f,0]
        fi = 0
        for c in range(nclosure):
            if cell_closures[cell, c] == facets[f]:
                facet_local_num[f,0] = fi
            if fStart <= cell_closures[cell, c] < fEnd:
                fi += 1

        # Second cell
        if facet_cells.shape[1] > 1:
            cell = facet_cells[f,1]
            if cell >= 0:
                fi = 0
                for c in range(nclosure):
                    if cell_closures[cell, c] == facets[f]:
                        facet_local_num[f,1] = fi
                    if fStart <= cell_closures[cell, c] < fEnd:
                        fi += 1
            else:
                facet_local_num[f,1] = -1
    return facet_local_num, facet_cells

@cython.boundscheck(False)
@cython.wraparound(False)
def closure_ordering(PETSc.DM plex,
                     PETSc.Section vertex_numbering,
                     PETSc.Section cell_numbering,
                     np.ndarray[np.int32_t, ndim=1, mode="c"] entity_per_cell):
    """Apply Fenics local numbering to a cell closure.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_numbering: Section describing the global cell numbering
    :arg entity_per_cell: List of the number of entity points in each dimension

    Vertices    := Ordered according to global/universal
                   vertex numbering
    Edges/faces := Ordered according to lexicographical
                   ordering of non-incident vertices
    """
    cdef:
        PetscInt c, cStart, cEnd, v, vStart, vEnd
        PetscInt f, fStart, fEnd, e, eStart, eEnd
        PetscInt dim, vi, ci, fi, v_per_cell, cell
        PetscInt offset, cell_offset, nfaces, nfacets
        PetscInt nclosure, nfacet_closure, nface_vertices
        PetscInt *vertices = NULL
        PetscInt *v_global = NULL
        PetscInt *closure = NULL
        PetscInt *facets = NULL
        PetscInt *faces = NULL
        PetscInt *face_indices = NULL
        PetscInt *face_vertices = NULL
        PetscInt *facet_vertices = NULL
        np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closure

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    fStart, fEnd = plex.getHeightStratum(1)
    eStart, eEnd = plex.getDepthStratum(1)
    vStart, vEnd = plex.getDepthStratum(0)
    v_per_cell = entity_per_cell[0]
    cell_offset = sum(entity_per_cell) - 1

    CHKERR(PetscMalloc1(v_per_cell, &vertices))
    CHKERR(PetscMalloc1(v_per_cell, &v_global))
    CHKERR(PetscMalloc1(v_per_cell, &facets))
    CHKERR(PetscMalloc1(v_per_cell-1, &facet_vertices))
    CHKERR(PetscMalloc1(entity_per_cell[1], &faces))
    CHKERR(PetscMalloc1(entity_per_cell[1], &face_indices))
    cell_closure = np.empty((cEnd - cStart, sum(entity_per_cell)), dtype=np.int32)

    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, c, PETSC_TRUE,
                                          &nclosure,&closure))

        # Find vertices and translate universal numbers
        vi = 0
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                vertices[vi] = closure[2*ci]
                CHKERR(PetscSectionGetOffset(vertex_numbering.sec,
                                             closure[2*ci], &v))
                # Correct -ve offsets for non-owned entities
                if v >= 0:
                    v_global[vi] = v
                else:
                    v_global[vi] = -(v+1)
                vi += 1

        # Sort vertices by universal number
        CHKERR(PetscSortIntWithArray(v_per_cell,v_global,vertices))
        for vi in range(v_per_cell):
            if dim == 1:
                # Correct 1D edge numbering
                cell_closure[cell, vi] = vertices[v_per_cell-vi-1]
            else:
                cell_closure[cell, vi] = vertices[vi]
        offset = v_per_cell

        # Find all faces (dim=1)
        if dim > 2:
            nfaces = 0
            for ci in range(nclosure):
                if eStart <= closure[2*ci] < eEnd:
                    faces[nfaces] = closure[2*ci]

                    CHKERR(DMPlexGetConeSize(plex.dm, closure[2*ci],
                                             &nface_vertices))
                    CHKERR(DMPlexGetCone(plex.dm, closure[2*ci],
                                         &face_vertices))

                    # Faces in 3D are tricky because we need a
                    # lexicographical sort with two keys (the local
                    # numbers of the two non-incident vertices).

                    # Find non-incident vertices
                    fi = 0
                    face_indices[nfaces] = 0
                    for v in range(v_per_cell):
                        incident = 0
                        for vi in range(nface_vertices):
                            if cell_closure[cell,v] == face_vertices[vi]:
                                incident = 1
                                break
                        if incident == 0:
                            face_indices[nfaces] += v * 10**(1-fi)
                            fi += 1
                    nfaces += 1

            # Sort by local numbers of non-incident vertices
            CHKERR(PetscSortIntWithArray(entity_per_cell[1],
                                         face_indices, faces))
            for fi in range(nfaces):
                cell_closure[cell, offset+fi] = faces[fi]
            offset += nfaces

        # Calling DMPlexGetTransitiveClosure() again invalidates the
        # current work array, so we need to get the facets and cell
        # out before getting the facet closures.

        # Find all facets (co-dim=1)
        nfacets = 0
        for ci in range(nclosure):
            if fStart <= closure[2*ci] < fEnd:
                facets[nfacets] = closure[2*ci]
                nfacets += 1

        # The cell itself is always the first entry in the Plex closure
        cell_closure[cell, cell_offset] = closure[0]

        # Now we can deal with facets
        if dim > 1:
            for f in range(nfacets):
                # Derive facet vertices from facet_closure
                CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                                  PETSC_TRUE,
                                                  &nfacet_closure,
                                                  &closure))
                vi = 0
                for fi in range(nfacet_closure):
                    if vStart <= closure[2*fi] < vEnd:
                        facet_vertices[vi] = closure[2*fi]
                        vi += 1

                # Find non-incident vertices
                for v in range(v_per_cell):
                    incident = 0
                    for vi in range(v_per_cell-1):
                        if cell_closure[cell,v] == facet_vertices[vi]:
                            incident = 1
                            break
                    # Only one non-incident vertex per facet, so
                    # local facet no. = non-incident vertex no.
                    if incident == 0:
                        cell_closure[cell,offset+v] = facets[f]
                        break

            offset += nfacets

    if closure != NULL:
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
                                              NULL, &closure))
    CHKERR(PetscFree(vertices))
    CHKERR(PetscFree(v_global))
    CHKERR(PetscFree(facets))
    CHKERR(PetscFree(facet_vertices))
    CHKERR(PetscFree(faces))
    CHKERR(PetscFree(face_indices))

    return cell_closure

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def quadrilateral_closure_ordering(PETSc.DM plex,
                                   PETSc.Section vertex_numbering,
                                   PETSc.Section cell_numbering,
                                   np.ndarray[np.int32_t, ndim=1, mode="c"] cell_orientations):
    """Cellwise orders mesh entities according to the given cell orientations.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_numbering: Section describing the cell numbering
    :arg cell_orientations: Specifies the starting vertex for each cell,
                            and the order of traversal (CCW or CW).
    """
    cdef:
        PetscInt c, cStart, cEnd, cell
        PetscInt fStart, fEnd, vStart, vEnd
        PetscInt entity_per_cell, ncells
        PetscInt nclosure, p, vi, v, fi, i
        PetscInt start_v, off
        PetscInt *closure = NULL
        PetscInt c_vertices[4]
        PetscInt c_facets[4]
        PetscInt g_vertices[4]
        PetscInt vertices[4]
        PetscInt facets[4]
        int reverse
        np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closure

    cStart, cEnd = plex.getHeightStratum(0)
    fStart, fEnd = plex.getHeightStratum(1)
    vStart, vEnd = plex.getDepthStratum(0)

    ncells = cEnd - cStart
    entity_per_cell = 4 + 4 + 1

    cell_closure = np.empty((ncells, entity_per_cell), dtype=np.int32)
    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, c, PETSC_TRUE, &nclosure, &closure))

        # First extract the facets (edges) and the vertices
        # from the transitive closure into c_facets and c_vertices.
        # Here we assume that DMPlex gives entities in the order:
        #
        #   8--3--7
        #   |     |
        #   4  0  2
        #   |     |
        #   5--1--6
        #
        # where the starting vertex and order of traversal is arbitrary.
        # (We fix that later.)
        #
        # For the vertices, we also retrieve the global numbers into g_vertices.
        vi = 0
        fi = 0
        for p in range(nclosure):
            if vStart <= closure[2*p] < vEnd:
                CHKERR(PetscSectionGetOffset(vertex_numbering.sec, closure[2*p], &v))
                c_vertices[vi] = closure[2*p]
                g_vertices[vi] = cabs(v)
                vi += 1
            elif fStart <= closure[2*p] < fEnd:
                c_facets[fi] = closure[2*p]
                fi += 1

        # The first vertex is given by the entry in cell_orientations.
        # The second vertex is always the one with the smaller global number.
        start_v = cell_orientations[cell]

        # Based on the cell orientation, we reorder the vertices and facets
        # (edges) from 'c_vertices' and 'c_facets' into 'vertices' and 'facets'.
        off = 0
        while off < 4 and g_vertices[off] != start_v:
            off += 1
        assert off < 4

        if g_vertices[(off + 1) % 4] < g_vertices[(off + 3) % 4]:
            for i in range(off, 4):
                vertices[i - off] = c_vertices[i]
                facets[i - off] = c_facets[i]
            for i in range(0, off):
                vertices[i + (4-off)] = c_vertices[i]
                facets[i + (4-off)] = c_facets[i]
        else:
            for i in range(off, -1, -1):
                vertices[off - i] = c_vertices[i]
            for i in range(3, off, -1):
                vertices[off+1 + (3-i)] = c_vertices[i]
            for i in range(off-1, -1, -1):
                facets[off-1 - i] = c_facets[i]
            for i in range(3, off-1, -1):
                facets[off + (3-i)] = c_facets[i]

        # At this point the cell "has" the right starting vertex
        # and order of traversal. If the starting vertex is one with an X,
        # and arrows on the edges show the order of traversal:
        #
        #   o--<--o
        #   |     |
        #   v     ^
        #   |     |
        #   o-->--X
        #
        # then outer product elements assume edge directions like this:
        #
        #   o--<--o
        #   |     |
        #   ^     ^
        #   |     |
        #   o--<--X
        #
        # ... and a vertex ordering like this:
        #
        #   3-----1
        #   |     |
        #   |     |
        #   |     |
        #   2-----0
        #
        # ... and a facet (edge) ordering like this:
        #
        #   o--3--o
        #   |     |
        #   1     0
        #   |     |
        #   o--2--o
        #
        # So let us permute.
        cell_closure[cell, 0] = vertices[0]
        cell_closure[cell, 1] = vertices[1]
        cell_closure[cell, 2] = vertices[3]
        cell_closure[cell, 3] = vertices[2]
        cell_closure[cell, 4 + 0] = facets[0]
        cell_closure[cell, 4 + 1] = facets[2]
        cell_closure[cell, 4 + 2] = facets[3]
        cell_closure[cell, 4 + 3] = facets[1]
        cell_closure[cell, 8] = c

    if closure != NULL:
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE, NULL, &closure))

    return cell_closure

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_nodes(PETSc.Section global_numbering,
                   np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closures,
                   fiat_element):
    """
    Builds the DoF mapping.

    :arg global_numbering: Section describing the global DoF numbering
    :arg cell_closures: 2D array of ordered cell closures
    :arg fiat_element: The FIAT element for the cell

    Preconditions: This function assumes that cell_closures contains mesh
    entities ordered by dimension, i.e. vertices first, then edges, faces, and
    finally the cell. For quadrilateral meshes, edges corresponding to
    dimension (0, 1) in the FIAT element must precede edges corresponding to
    dimension (1, 0) in the FIAT element.
    """
    cdef:
        int *ceil_ndofs = NULL
        int *flat_index = NULL
        PetscInt ncells, nclosure, dofs_per_cell
        PetscInt c, i, j, k
        PetscInt entity, ndofs, off
        np.ndarray[np.int32_t, ndim=2, mode="c"] cell_nodes

    ncells = cell_closures.shape[0]
    nclosure = cell_closures.shape[1]

    # Extract ordering from FIAT element entity DoFs
    ndofs_list = []
    flat_index_list = []

    entity_dofs = fiat_element.entity_dofs()
    for dim in sorted(entity_dofs.keys()):
        for entity_num in xrange(len(entity_dofs[dim])):
            dofs = entity_dofs[dim][entity_num]

            ndofs_list.append(len(dofs))
            flat_index_list.extend(dofs)

    # Coerce lists into C arrays
    assert nclosure == len(ndofs_list)
    dofs_per_cell = len(flat_index_list)

    CHKERR(PetscMalloc1(nclosure, &ceil_ndofs))
    CHKERR(PetscMalloc1(dofs_per_cell, &flat_index))

    for i in range(nclosure):
        ceil_ndofs[i] = ndofs_list[i]
    for i in range(dofs_per_cell):
        flat_index[i] = flat_index_list[i]

    # Fill cell nodes
    cell_nodes = np.empty((ncells, dofs_per_cell), dtype=np.int32)
    for c in range(ncells):
        k = 0
        for i in range(nclosure):
            entity = cell_closures[c, i]
            CHKERR(PetscSectionGetDof(global_numbering.sec, entity, &ndofs))
            if ndofs > 0:
                CHKERR(PetscSectionGetOffset(global_numbering.sec, entity, &off))
                for j in range(ceil_ndofs[i]):
                    cell_nodes[c, flat_index[k]] = off + j
                    k += 1

    CHKERR(PetscFree(ceil_ndofs))
    CHKERR(PetscFree(flat_index))
    return cell_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def get_facet_nodes(np.ndarray[np.int32_t, ndim=2, mode="c"] facet_cells,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] cell_nodes):
    """
    Derives the DoF mapping for a given facet list.

    :arg facet_cells: 2D array of parent cells for each facet
    :arg cell_nodes: 2D array of cell DoFs
    """
    cdef:
        int f, i, cell, nfacets, ncells, ndofs
        np.ndarray[np.int32_t, ndim=2, mode="c"] facet_nodes

    nfacets = facet_cells.shape[0]
    ncells = facet_cells.shape[1]
    ndofs = cell_nodes.shape[1]
    facet_nodes = np.empty((nfacets, ncells*ndofs), dtype=np.int32)

    for f in range(nfacets):
        # First parent cell
        cell = facet_cells[f, 0]
        for i in range(ndofs):
            facet_nodes[f, i] = cell_nodes[cell, i]

        # Second parent cell for internal facets
        if ncells > 1:
            cell = facet_cells[f, 1]
            if cell >= 0:
                for i in range(ndofs):
                    facet_nodes[f, ndofs+i] = cell_nodes[cell, i]
            else:
                for i in range(ndofs):
                    facet_nodes[f, ndofs+i] = -1

    return facet_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def label_facets(PETSc.DM plex, label_boundary=True):
    """Add labels to facets in the the plex

    Facets on the boundary are marked with "exterior_facets" while all
    others are marked with "interior_facets".

    :arg label_boundary: if False, don't label the boundary faces
         (they must have already been labelled)."""
    cdef:
        PetscInt fStart, fEnd, facet, val
        char *ext_label = <char *>"exterior_facets"
        char *int_label = <char *>"interior_facets"

    # Mark boundaries as exterior_facets
    if label_boundary:
        plex.markBoundaryFaces(ext_label)
    plex.createLabel(int_label)

    fStart, fEnd = plex.getHeightStratum(1)

    for facet in range(fStart, fEnd):
        CHKERR(DMPlexGetLabelValue(plex.dm, ext_label, facet, &val))
        # Not marked, must be interior
        if val == -1:
            CHKERR(DMPlexSetLabelValue(plex.dm, int_label, facet, 1))

@cython.boundscheck(False)
@cython.wraparound(False)
def reordered_coords(PETSc.DM plex, PETSc.Section global_numbering, shape):
    """Return coordinates for the plex, reordered according to the
    global numbering permutation for the coordinate function space.

    Shape is a tuple of (plex.numVertices(), geometric_dim)."""
    cdef:
        PetscInt v, vStart, vEnd, offset
        PetscInt i, dim = shape[1]
        np.ndarray[np.float64_t, ndim=2, mode="c"] plex_coords, coords

    plex_coords = plex.getCoordinatesLocal().array.reshape(shape)
    coords = np.empty_like(plex_coords)
    vStart, vEnd = plex.getDepthStratum(0)

    for v in range(vStart, vEnd):
        CHKERR(PetscSectionGetOffset(global_numbering.sec, v, &offset))
        for i in range(dim):
            coords[offset, i] = plex_coords[v - vStart, i]

    return coords

@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes(PETSc.DM plex):
    """Mark all points in a given Plex according to the PyOP2 entity
    classes:

    core      : owned and not in send halo
    non_core  : owned and in send halo
    exec_halo : in halo, but touch owned entity
    non_exec_halo : in halo and only touch halo entities

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt p, pStart, pEnd, cStart, cEnd, vStart, vEnd
        PetscInt c, ncells, f, nfacets, ci, nclosure, vi, dim
        PetscInt depth, non_core, exec_halo, nroots, nleaves
        PetscInt v_per_cell
        PetscInt *cells = NULL
        PetscInt *facets = NULL
        PetscInt *vertices = NULL
        PetscInt *closure = NULL
        PetscInt *ilocal = NULL
        PetscBool non_exec
        PetscSFNode *iremote = NULL
        PETSc.SF point_sf = None
        PETSc.IS cell_is = None
        PETSc.IS facet_is = None

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    v_per_cell = plex.getConeSize(cStart)
    CHKERR(PetscMalloc1(v_per_cell, &vertices))

    plex.createLabel("op2_core")
    plex.createLabel("op2_non_core")
    plex.createLabel("op2_exec_halo")
    plex.createLabel("op2_non_exec_halo")

    lbl_depth = <char*>"depth"
    lbl_core = <char*>"op2_core"
    lbl_non_core = <char*>"op2_non_core"
    lbl_halo = <char*>"op2_exec_halo"
    lbl_non_exec_halo = <char*>"op2_non_exec_halo"

    if MPI.comm.size > 1:
        # Mark exec_halo from point overlap SF
        point_sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(point_sf.sf, &nroots, &nleaves,
                               &ilocal, &iremote))
        for p in range(nleaves):
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       ilocal[p], &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_halo,
                                       ilocal[p], depth))
    else:
        # If sequential mark all points as core
        pStart, pEnd = plex.getChart()
        for p in range(pStart, pEnd):
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       p, &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_core,
                                       p, depth))
        CHKERR(PetscFree(vertices))
        return

    # Mark all cells adjacent to halo cells as non_core,
    # where adjacent(c) := star(closure(c))
    ncells = plex.getStratumSize("op2_exec_halo", dim)
    cell_is = plex.getStratumIS("op2_exec_halo", dim)
    CHKERR(ISGetIndices(cell_is.iset, &cells))
    for c in range(ncells):
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                          PETSC_TRUE,
                                          &nclosure,
                                          &closure))
        # Copy vertices out of the work array (closure)
        vi = 0
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                vertices[vi] = closure[2*ci]
                vi += 1

        # Mark all cells in the star of each vertex
        for vi in range(v_per_cell):
            vertex = vertices[vi]
            CHKERR(DMPlexGetTransitiveClosure(plex.dm, vertices[vi],
                                              PETSC_FALSE,
                                              &nclosure,
                                              &closure))
            for ci in range(nclosure):
                if cStart <= closure[2*ci] < cEnd:
                    p = closure[2*ci]
                    CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                               p, &exec_halo))
                    if exec_halo < 0:
                        CHKERR(DMPlexSetLabelValue(plex.dm,
                                                   lbl_non_core,
                                                   p, dim))

    # Mark the closures of non_core cells as non_core
    ncells = plex.getStratumSize("op2_non_core", dim)
    cell_is = plex.getStratumIS("op2_non_core", dim)
    CHKERR(ISGetIndices(cell_is.iset, &cells))
    for c in range(ncells):
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                          PETSC_TRUE,
                                          &nclosure,
                                          &closure))
        for ci in range(nclosure):
            p = closure[2*ci]
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                       p, &exec_halo))
            if exec_halo < 0:
                CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                           p, &depth))
                CHKERR(DMPlexSetLabelValue(plex.dm, lbl_non_core,
                                           p, depth))

    # Mark all remaining points as core
    pStart, pEnd = plex.getChart()
    for p in range(pStart, pEnd):
        CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                   p, &exec_halo))
        CHKERR(DMPlexGetLabelValue(plex.dm, lbl_non_core,
                                   p, &non_core))
        if exec_halo < 0 and non_core < 0:
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       p, &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_core,
                                       p, depth))

    # Halo facets that only touch halo vertices and halo cells need to
    # be marked as non-exec.
    nfacets = plex.getStratumSize("op2_exec_halo", dim-1)
    facet_is = plex.getStratumIS("op2_exec_halo", dim-1)
    CHKERR(ISGetIndices(facet_is.iset, &facets))
    for f in range(nfacets):
        non_exec = PETSC_TRUE
        # Check for halo vertices
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                          PETSC_TRUE,
                                          &nclosure,
                                          &closure))
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                           closure[2*ci], &exec_halo))
                if exec_halo < 0:
                    # Touches a non-halo vertex, needs to be executed
                    # over.
                    non_exec = PETSC_FALSE
        if non_exec:
            # If we still think we're non-exec, check for halo cells
            CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                              PETSC_FALSE,
                                              &nclosure,
                                              &closure))
            for ci in range(nclosure):
                if cStart <= closure[2*ci] < cEnd:
                    CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                               closure[2*ci], &exec_halo))
                    if exec_halo < 0:
                        # Touches a non-halo cell, needs to be
                        # executed over.
                        non_exec = PETSC_FALSE
        if non_exec:
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       facets[f], &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_non_exec_halo,
                                       facets[f], depth))
            # Remove facet from exec-halo label
            CHKERR(DMPlexClearLabelValue(plex.dm, lbl_halo,
                                         facets[f], depth))

    if closure != NULL:
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
                                              NULL, &closure))
    CHKERR(PetscFree(vertices))

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_classes(PETSc.DM plex):
    """Builds PyOP2 entity class offsets of cells.

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt dim, c, ci, nclass

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    cell_classes = [0, 0, 0, 0]
    c = 0

    for i, op2class in enumerate(["op2_core",
                                  "op2_non_core",
                                  "op2_exec_halo"]):
        c += plex.getStratumSize(op2class, dim)
        cell_classes[i] = c

    cell_classes[3] = cell_classes[2]
    return cell_classes

@cython.boundscheck(False)
@cython.wraparound(False)
def get_facets_by_class(PETSc.DM plex, label):
    """Builds a list of all facets ordered according to OP2 entity
    classes and computes the respective class offsets.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg label: Label string that marks the facets to order
    """
    cdef:
        PetscInt dim, fi, ci, nfacets, nclass, lbl_val
        PetscInt *indices = NULL
        PETSc.IS class_is = None
        char *class_chr = NULL
        np.ndarray[np.int32_t, ndim=1, mode="c"] facets

    label_chr = <char*>label
    dim = plex.getDimension()
    nfacets = plex.getStratumSize(label, 1)
    facets = np.empty(nfacets, dtype=np.int32)
    facet_classes = [0, 0, 0, 0]
    fi = 0

    for i, op2class in enumerate(["op2_core",
                                  "op2_non_core",
                                  "op2_exec_halo",
                                  "op2_non_exec_halo"]):
        nclass = plex.getStratumSize(op2class, dim-1)
        if nclass > 0:
            class_is = plex.getStratumIS(op2class, dim-1)
            CHKERR(ISGetIndices(class_is.iset, &indices))
            for ci in range(nclass):
                CHKERR(DMPlexGetLabelValue(plex.dm, label_chr,
                                           indices[ci], &lbl_val))
                if lbl_val == 1:
                    facets[fi] = indices[ci]
                    fi += 1
        facet_classes[i] = fi

    return facets, facet_classes

@cython.boundscheck(False)
@cython.wraparound(False)
def plex_renumbering(PETSc.DM plex, np.ndarray[PetscInt, ndim=1, mode="c"] reordering=None):
    """
    Build a global node renumbering as a permutation of Plex points.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg reordering: A reordering from reordered to original plex
         points used to provide the traversal order of the cells
         (i.e. the inverse of the ordering obtained from
         DMPlexGetOrdering).  Optional, if not provided (or ``None``),
         no reordering is applied and the plex is traversed in
         original order.

    The node permutation is derived from a depth-first traversal of
    the Plex graph over each OP2 entity class in turn. The returned IS
    is the Plex -> OP2 permutation.
    """
    cdef:
        PetscInt dim, ncells, nfacets, nclosure, c, ci, p, p_glbl, lbl_val
        PetscInt cStart, cEnd, core_idx, non_core_idx, exec_halo_idx
        PetscInt *core_cells = NULL
        PetscInt *non_core_cells = NULL
        PetscInt *exec_halo_cells = NULL
        PetscInt *cells = NULL
        PetscInt *facets = NULL
        PetscInt *closure = NULL
        PetscInt *perm = NULL
        PETSc.IS cell_is = None
        PETSc.IS facet_is = None
        PETSc.IS perm_is = None
        char *lbl_chr = NULL
        PetscBT seen = NULL
        bint reorder = reordering is not None

    dim = plex.getDimension()
    pStart, pEnd = plex.getChart()
    cStart, cEnd = plex.getHeightStratum(0)
    CHKERR(PetscMalloc1(pEnd - pStart, &perm))
    CHKERR(PetscBTCreate(pEnd - pStart, &seen))
    p_glbl = 0

    if reorder:
        core_idx = plex.getStratumSize("op2_core", dim)
        non_core_idx = plex.getStratumSize("op2_non_core", dim)
        exec_halo_idx = plex.getStratumSize("op2_exec_halo", dim)

        CHKERR(PetscMalloc1(core_idx, &core_cells))
        CHKERR(PetscMalloc1(non_core_idx, &non_core_cells))
        CHKERR(PetscMalloc1(exec_halo_idx, &exec_halo_cells))

        core_idx = 0
        non_core_idx = 0
        exec_halo_idx = 0

        # Walk over the reordering
        for ci in range(reordering.size):
            # Have we hit all the cells yet, if so break out early
            if core_idx + non_core_idx + exec_halo_idx > cEnd - cStart:
                break
            p = reordering[ci]

            CHKERR(DMPlexGetLabelValue(plex.dm, "depth",
                                       p, &lbl_val))
            # This point is a cell
            if lbl_val == dim:
                # Which entity class is this point in?
                CHKERR(DMPlexGetLabelValue(plex.dm, "op2_core",
                                           p, &lbl_val))
                if lbl_val == dim:
                    core_cells[core_idx] = p
                    core_idx += 1
                    continue

                CHKERR(DMPlexGetLabelValue(plex.dm, "op2_non_core",
                                           p, &lbl_val))
                if lbl_val == dim:
                    non_core_cells[non_core_idx] = p
                    non_core_idx += 1
                    continue

                CHKERR(DMPlexGetLabelValue(plex.dm, "op2_exec_halo",
                                           p, &lbl_val))

                if lbl_val == dim:
                    exec_halo_cells[exec_halo_idx] = p
                    exec_halo_idx += 1
                    continue

                raise RuntimeError("Should never be reached")

    # Now we can walk over the cell classes and order all the plex points
    for op2class in ["op2_core", "op2_non_core", "op2_exec_halo"]:
        lbl_chr = <char *>op2class
        if reorder:
            if op2class == "op2_core":
                ncells = core_idx
                cells = core_cells
            elif op2class == "op2_non_core":
                ncells = non_core_idx
                cells = non_core_cells
            elif op2class == "op2_exec_halo":
                ncells = exec_halo_idx
                cells = exec_halo_cells
        else:
            ncells = plex.getStratumSize(op2class, dim)
            if ncells > 0:
                cell_is = plex.getStratumIS(op2class, dim)
                CHKERR(ISGetIndices(cell_is.iset, &cells))
        for c in range(ncells):
            CHKERR(DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                              PETSC_TRUE,
                                              &nclosure,
                                              &closure))
            for ci in range(nclosure):
                p = closure[2*ci]
                if not PetscBTLookup(seen, p):
                    CHKERR(DMPlexGetLabelValue(plex.dm, lbl_chr,
                                               p, &lbl_val))
                    if lbl_val >= 0:
                        CHKERR(PetscBTSet(seen, p))
                        perm[p_glbl] = p
                        p_glbl += 1

        if not reorder and ncells > 0:
            CHKERR(ISRestoreIndices(cell_is.iset, &cells))

    if closure != NULL:
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
                                              NULL, &closure))

    # We currently mark non-exec facets without marking non-exec
    # cells, so they will not get picked up by the cell closure loops
    # and we need to add them explicitly.
    op2class = "op2_non_exec_halo"
    nfacets = plex.getStratumSize(op2class, dim-1)
    if nfacets > 0:
        facet_is = plex.getStratumIS(op2class, dim-1)
        CHKERR(ISGetIndices(facet_is.iset, &facets))
        for f in range(nfacets):
            p = facets[f]
            if not PetscBTLookup(seen, p):
                CHKERR(PetscBTSet(seen, p))
                perm[p_glbl] = p
                p_glbl += 1
    if reorder:
        CHKERR(PetscFree(core_cells))
        CHKERR(PetscFree(non_core_cells))
        CHKERR(PetscFree(exec_halo_cells))

    CHKERR(PetscBTDestroy(&seen))
    perm_is = PETSc.IS().create()
    perm_is.setType("general")
    CHKERR(ISGeneralSetIndices(perm_is.iset, pEnd - pStart,
                               perm, PETSC_OWN_POINTER))
    return perm_is

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_remote_ranks(PETSc.DM plex):
    """Returns an array assigning the rank of the owner to each
    locally visible cell. Locally owned cells have -1 assigned to them.

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt cStart, cEnd, ncells, i
        PETSc.SF sf
        PetscInt nroots, nleaves
        PetscInt *ilocal
        PetscSFNode *iremote
        np.ndarray[np.int32_t, ndim=1, mode="c"] result

    cStart, cEnd = plex.getHeightStratum(0)
    ncells = cEnd - cStart

    result = np.full(ncells, -1, dtype=np.int32)
    if MPI.comm.size > 1:
        sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, &nroots, &nleaves, &ilocal, &iremote))

        for i in range(nleaves):
            if cStart <= ilocal[i] < cEnd:
                result[ilocal[i] - cStart] = iremote[i].rank

    return result

cdef inline PetscInt cneg(PetscInt i):
    """complementary inverse"""
    return -(i + 1)

cdef inline PetscInt cabs(PetscInt i):
    """complementary absolute value"""
    if i >= 0:
        return i
    else:
        return cneg(i)

cdef inline void get_edge_global_vertices(PETSc.DM plex,
                                          PETSc.Section vertex_numbering,
                                          PetscInt facet,
                                          PetscInt *global_v):
    """Returns the global numbers of the vertices of an edge.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg facet: The edge
    :arg global_v: Return buffer, must have capacity for 2 values
    """
    cdef:
        PetscInt nvertices, ndofs
        PetscInt *vs = NULL

    CHKERR(DMPlexGetConeSize(plex.dm, facet, &nvertices))
    assert nvertices == 2

    CHKERR(DMPlexGetCone(plex.dm, facet, &vs))

    CHKERR(PetscSectionGetDof(vertex_numbering.sec, vs[0], &ndofs))
    assert cabs(ndofs) == 1
    CHKERR(PetscSectionGetDof(vertex_numbering.sec, vs[1], &ndofs))
    assert cabs(ndofs) == 1

    CHKERR(PetscSectionGetOffset(vertex_numbering.sec, vs[0], &global_v[0]))
    CHKERR(PetscSectionGetOffset(vertex_numbering.sec, vs[1], &global_v[1]))

    global_v[0] = cabs(global_v[0])
    global_v[1] = cabs(global_v[1])

cdef inline np.int8_t get_global_edge_orientation(PETSc.DM plex,
                                                  PETSc.Section vertex_numbering,
                                                  PetscInt facet):
    """Returns the local plex direction (ordering in plex cone) relative to
    the global edge direction (from smaller to greater global vertex number).

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg facet: The edge
    """
    cdef PetscInt v[2]
    get_edge_global_vertices(plex, vertex_numbering, facet, v)
    return v[0] > v[1]

cdef struct CommFacet:
    PetscInt remote_rank
    PetscInt global_u, global_v
    PetscInt local_facet

cdef int CommFacet_cmp(void *x_, void *y_) nogil:
    """Three-way comparison C function for CommFacet structs."""
    cdef:
        CommFacet *x = <CommFacet *>x_
        CommFacet *y = <CommFacet *>y_

    if x.remote_rank < y.remote_rank:
        return -1
    elif x.remote_rank > y.remote_rank:
        return 1

    if x.global_u < y.global_u:
        return -1
    elif x.global_u > y.global_u:
        return 1

    if x.global_v < y.global_v:
        return -1
    elif x.global_v > y.global_v:
        return 1

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void get_communication_lists(
    PETSc.DM plex, PETSc.Section vertex_numbering,
    np.ndarray[np.int32_t, ndim=1, mode="c"] cell_ranks,
    # Output parameters:
    np.int32_t *nranks, np.int32_t **ranks, np.int32_t **offsets,
    np.int32_t **facets, np.int32_t **facet2index):

    """Creates communication lists for shared facet information exchange.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.

    :arg nranks: Number of neighbouring MPI nodes (return value)
    :arg ranks: MPI ranks of neigbours (return value)
    :arg offsets: Offset for each neighbour in the data buffer (return value)
    :arg facets: Array of local plex facet numbers of shared facets
                 (return value)
    :arg facet2index: Maps local facet numbers to indices in the communication
                      buffer, inverse of 'facets' (return value)
    """
    cdef:
        int comm_size = MPI.comm.size
        PetscInt cStart, cEnd
        PetscInt nfacets, fStart, fEnd, f
        PetscInt i, k, support_size
        PetscInt *support = NULL
        PetscInt local_count, remote
        PetscInt v[2]
        np.int32_t *facet_ranks = NULL
        np.int32_t *nfacets_per_rank = NULL

        CommFacet *cfacets = NULL

    cStart, cEnd = plex.getHeightStratum(0)
    fStart, fEnd = plex.getHeightStratum(1)
    nfacets = fEnd - fStart

    CHKERR(PetscMalloc1(nfacets, &facet_ranks))
    memset(facet_ranks, -1, nfacets * sizeof(np.int32_t))

    # Determines which facets are shared, and which MPI process
    # they are shared with.
    for f in range(fStart, fEnd):
        CHKERR(DMPlexGetSupportSize(plex.dm, f, &support_size))
        CHKERR(DMPlexGetSupport(plex.dm, f, &support))

        local_count = 0
        remote = -1
        for i in range(support_size):
            if cell_ranks[support[i] - cStart] >= 0:
                remote = cell_ranks[support[i] - cStart]
            else:
                local_count += 1

        if local_count == 1:
            facet_ranks[f - fStart] = remote

    # Counts how many facets are shared with each MPI node
    CHKERR(PetscMalloc1(comm_size, &nfacets_per_rank))
    memset(nfacets_per_rank, 0, comm_size * sizeof(np.int32_t))

    for i in range(nfacets):
        if facet_ranks[i] != -1:
            nfacets_per_rank[facet_ranks[i]] += 1

    # Counts how many MPI nodes shall this node communicate with
    nranks[0] = 0
    for i in range(comm_size):
        if nfacets_per_rank[i] != 0:
            nranks[0] += 1

    # Creates list of neighbours, and their offsets
    # in the communication buffer.
    #
    # Information about facets shared with rank 'i'
    # should be between offsets[i] (inclusive) and
    # offset[i+1] (exclusive) in the buffer.
    CHKERR(PetscMalloc1(nranks[0], ranks))
    CHKERR(PetscMalloc1(nranks[0]+1, offsets))

    offsets[0][0] = 0
    k = 0
    for i in range(comm_size):
        if nfacets_per_rank[i] != 0:
            ranks[0][k] = i
            offsets[0][k+1] = offsets[0][k] + nfacets_per_rank[i]
            k += 1

    CHKERR(PetscFree(nfacets_per_rank))

    # Sort the facets based on
    # 1. Remote rank - so they occupy the right section of the buffer.
    # 2. Global vertex numbers - so the same order is used on both sides.
    CHKERR(PetscMalloc1(offsets[0][nranks[0]], &cfacets))

    k = 0
    for f in range(fStart, fEnd):
        if facet_ranks[f - fStart] != -1:
            cfacets[k].remote_rank = facet_ranks[f - fStart]
            get_edge_global_vertices(plex, vertex_numbering, f, v)
            if v[0] < v[1]:
                cfacets[k].global_u = v[0]
                cfacets[k].global_v = v[1]
            else:
                cfacets[k].global_u = v[1]
                cfacets[k].global_v = v[0]
            cfacets[k].local_facet = f
            k += 1
    CHKERR(PetscFree(facet_ranks))
    qsort(cfacets, offsets[0][nranks[0]], sizeof(CommFacet), &CommFacet_cmp)

    # For debugging purposes:
    #
    # for i in range(offsets[0][nranks[0]]):
    #     print "(%d/%d): %d = (%d, %d) -> %d" % (MPI.comm.rank,
    #                                             MPI.comm.size,
    #                                             cfacets[i].local_facet,
    #                                             cfacets[i].global_u,
    #                                             cfacets[i].global_v,
    #                                             cfacets[i].remote_rank)

    CHKERR(PetscMalloc1(offsets[0][nranks[0]], facets))
    CHKERR(PetscMalloc1(nfacets, facet2index))
    memset(facet2index[0], -1, nfacets * sizeof(np.int32_t))

    for i in range(offsets[0][nranks[0]]):
        facets[0][i] = cfacets[i].local_facet
        facet2index[0][facets[0][i] - fStart] = i
    CHKERR(PetscFree(cfacets))

    # For debugging purposes:
    #
    # for i in range(nfacets):
    #     if facet2index[0][i] != -1:
    #         print "(%d/%d): [%d] = %d" % (MPI.comm.rank,
    #                                       MPI.comm.size,
    #                                       facet2index[0][i],
    #                                       fStart + i)

@cython.profile(False)
cdef inline void plex_get_restricted_support(PETSc.DM plex,
                                             np.int32_t *cell_ranks,
                                             PetscInt f,
                                             # Output parameters:
                                             PetscInt *size,
                                             PetscInt *outbuf):
    """Returns the owned cells incident to a given facet.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    :arg f: Facet, whose owned support is the result
    :arg size: Length of the result
    :arg outbuf: Preallocated output buffer
    """
    cdef:
        PetscInt cStart, cEnd, c
        PetscInt support_size
        PetscInt *support = NULL
        PetscInt i, k

    CHKERR(DMPlexGetHeightStratum(plex.dm, 0, &cStart, &cEnd))

    CHKERR(DMPlexGetSupportSize(plex.dm, f, &support_size))
    CHKERR(DMPlexGetSupport(plex.dm, f, &support))

    k = 0
    for i in range(support_size):
        if cell_ranks[support[i] - cStart] < 0:
            outbuf[k] = support[i]
            k += 1
    size[0] = k

@cython.cdivision(True)
cdef inline PetscInt traverse_cell_string(PETSc.DM plex,
                                          PetscInt first_facet,
                                          PetscInt cell,
                                          np.int32_t *cell_ranks,
                                          np.int8_t *orientations):
    """Takes a start facet, and a direction (which of the, possibly two, cells
    it is adjacent to) and propagates that facet's orientation as far as
    possible by orienting the "opposite" facet in the cell then moving to the
    next cell.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg first_facet: Facet to start traversal with
    :arg cell: One of the cells incident to 'first_facet', determines
               the direction of traversal.
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    :arg orientations: Facet orientations relative to the plex.
                       -1: orientation not set
                        0: orientation is same as in (local) plex
                       +1: orientation is the opposite of that in plex
                       THIS ARRAY IS UPDATED BY THIS FUNCTION.

    Returns the plex number of the facet visited last, or -1 if
    'first_facet' is part of a closed loop.

    Facets not incident to an owned cell are ignored.

    Facet orientations are aligned to match the orientation, which was
    assigned to orientations[first_facet - fStart].
    """
    cdef:
        PetscInt fStart, fEnd
        PetscInt from_facet = first_facet
        PetscInt to_facet
        PetscInt c = cell
        np.int8_t plex_orientation

        PetscInt local_from, local_to

        PetscInt cone_size, support_size
        PetscInt *cone = NULL
        PetscInt *cone_orient = NULL
        PetscInt support[2]
        PetscInt i, ncells_adj

    CHKERR(DMPlexGetHeightStratum(plex.dm, 1, &fStart, &fEnd))

    # Retrieve orientation of first facet
    plex_orientation = orientations[first_facet - fStart]

    while True:
        CHKERR(DMPlexGetConeSize(plex.dm, c, &cone_size))
        assert cone_size == 4

        CHKERR(DMPlexGetCone(plex.dm, c, &cone))
        local_from = 0
        while cone[local_from] != from_facet and local_from < cone_size:
            local_from += 1
        assert local_from < cone_size

        local_to = (local_from + 2) % 4
        to_facet = cone[local_to]

        CHKERR(DMPlexGetConeOrientation(plex.dm, c, &cone_orient))
        plex_orientation ^= (cone_orient[local_from] < 0) ^ True ^ (cone_orient[local_to] < 0)

        # Store orientation of next facet
        orientations[to_facet - fStart] = plex_orientation

        if to_facet == first_facet:
            # Closed loop
            return -1

        plex_get_restricted_support(plex, cell_ranks, to_facet, &support_size, support)

        ncells_adj = 0
        for i in range(support_size):
            if support[i] != c:
                ncells_adj += 1

        if ncells_adj == 0:
            # Reached boundary of local domain
            return to_facet
        elif ncells_adj == 1:
            # Continue with next cell
            for i in range(support_size):
                if support[i] != c:
                    from_facet = to_facet
                    c = support[i]
                    break
        else:
            assert ncells_adj > 1
            raise RuntimeError("Facet belongs to more than two quadrilaterals!")

    # We must not reach this point here.
    raise RuntimeError("This should never happen!")

@cython.boundscheck(False)
@cython.wraparound(False)
cdef locally_orient_quadrilateral_plex(PETSc.DM plex,
                                       PETSc.Section vertex_numbering,
                                       np.int32_t *cell_ranks,
                                       np.int32_t *facet2index,
                                       np.int32_t nfacets_shared,
                                       np.int8_t *orientations):
    """Locally orient the facets (edges) of a quadrilateral plex, and
    derive the dependency information of shared facets.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    :arg facet2index: Maps plex facet numbers to their index in the buffer
                      of shared facets.
    :arg nfacets_shared: Number of facets shared with other MPI nodes.
    :arg orientations: Facet orientations relative to the plex.
                       -1: orientation not set
                        0: orientation is same as in (local) plex
                       +1: orientation is the opposite of that in plex
                       THIS ARRAY IS UPDATED BY THIS FUNCTION.

    Returns an array of size 'nfacets_shared', which tells for each shared
    facet which other shared facet needs update, if any, when a shared facet
    is flipped.
     * Equal to 'nfacets_shared': no other facet requires update.
     * Non-negative value: index of shared facet,
                           which must have the same global orientation.
     * Negative value 'i': cneg(i) is the index of the shared facet,
                           which must have opposite global orientation.
    """
    cdef:
        PetscInt nfacets, fStart, fEnd, f
        PetscInt size
        PetscInt support[2]
        PetscInt start_facet, end_facet
        np.int8_t twist
        PetscInt i, j
        np.ndarray[np.int32_t, ndim=1, mode="c"] result

    fStart, fEnd = plex.getHeightStratum(1)
    nfacets = fEnd - fStart

    result = np.empty(nfacets_shared, dtype=np.int32)

    # Here we walk over all the known facets, if it is not oriented already.
    for f in range(fStart, fEnd):
        if orientations[f - fStart] < 0:
            orientations[f - fStart] = 0

            plex_get_restricted_support(plex, cell_ranks, f, &size, support)
            assert 0 <= size <= 2
            if size == 0:
                # Facet is interior to some other MPI process, ignored
                continue

            # Propagate the orientation of this facet as far as possible
            end_facet = traverse_cell_string(plex, f, support[0],
                                             cell_ranks, orientations)
            if end_facet == -1:
                # Closed loop
                if orientations[f - fStart]:
                    # Moebius strip found
                    #
                    # So we came round a loop and found that the last cell we
                    # hit claims that the end_facet must be flipped then there
                    # must be a twist in the loop, because the first facet
                    # (which is the same) should have had no flip.
                    raise RuntimeError("Moebius strip found in the mesh.")
            else:
                if size == 1:
                    # 'f' is at local domain boundary
                    start_facet = f
                else:
                    # Here we potentially walk off in the other direction
                    start_facet = traverse_cell_string(plex, f, support[1],
                                                       cell_ranks, orientations)

                i = facet2index[start_facet - fStart]
                j = facet2index[end_facet - fStart]
                if i >= 0 or j >= 0:
                    # Either the start or the end facet is shared
                    # with remote processes
                    twist = 0
                    twist ^= get_global_edge_orientation(plex,
                                                         vertex_numbering,
                                                         start_facet)
                    twist ^= orientations[start_facet - fStart]
                    twist ^= orientations[end_facet - fStart]
                    twist ^= get_global_edge_orientation(plex,
                                                         vertex_numbering,
                                                         end_facet)

                    # If the other end of the string is local (not shared), then
                    # no propagation to remote ranks at the other end is needed.
                    if i == -1:
                        result[j] = nfacets_shared
                    elif j == -1:
                        result[i] = nfacets_shared
                    # If other end of the string is shared, then propagation
                    # must take place, the sign tells you whether an orientation
                    # flip is required and the value tells you which facet.
                    elif twist == 0:
                        result[i] = j
                        result[j] = i
                    else:
                        result[i] = cneg(j)
                        result[j] = cneg(i)

    # At the end of this function we have provided a consistent orientation
    # to the local plex, in O(nfacets) time, and we are returning, for all
    # shared facets, information about whether they will require a round of
    # communications when we try and provide a globally consistent orientation.
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void exchange_edge_orientation_data(
    np.int32_t nranks, np.int32_t *ranks, np.int32_t *offsets,
    np.ndarray[np.int32_t, ndim=1, mode="c"] ours,
    np.ndarray[np.int32_t, ndim=1, mode="c"] theirs):

    """Exchange edge orientation data between neighbouring MPI nodes.

    :arg nranks: Number of neighbouring MPI nodes
    :arg ranks: MPI ranks of neigbours
    :arg offsets: Offset for each neighbour in the data buffer
    :arg ours: Local data, to be sent to neigbours
    :arg theirs: Remote data, to be received from neighbours (return value)
    """
    cdef np.int32_t ri

    # Initiate receiving
    recv_reqs = []
    for ri in range(nranks):
        recv_reqs.append(MPI.comm.Irecv(theirs[offsets[ri] : offsets[ri+1]], ranks[ri]))

    # Initiate sending
    send_reqs = []
    for ri in range(nranks):
        send_reqs.append(MPI.comm.Isend(ours[offsets[ri] : offsets[ri+1]], ranks[ri]))

    # Wait for completion
    for req in recv_reqs:
        req.Wait()
    for req in send_reqs:
        req.Wait()

@cython.boundscheck(False)
@cython.wraparound(False)
def quadrilateral_facet_orientations(
    PETSc.DM plex, PETSc.Section vertex_numbering,
    np.ndarray[np.int32_t, ndim=1, mode="c"] cell_ranks):

    """Returns globally synchronised facet orientations (edge directions)
    incident to locally owned quadrilateral cells.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    """
    cdef:
        np.int32_t nranks
        np.int32_t *ranks = NULL
        np.int32_t *offsets = NULL
        np.int32_t *facets = NULL
        np.int32_t *facet2index = NULL

        PetscInt nfacets, nfacets_shared, fStart, fEnd

        np.ndarray[np.int32_t, ndim=1, mode="c"] affects
        np.ndarray[np.int32_t, ndim=1, mode="c"] ours, theirs
        np.int32_t conflict, value, f, i, j

        PetscInt ci, size
        PetscInt cells[2]

        np.ndarray[np.int8_t, ndim=1, mode="c"] result

    # Get communication lists
    get_communication_lists(plex, vertex_numbering, cell_ranks,
                            &nranks, &ranks, &offsets, &facets, &facet2index)
    nfacets_shared = offsets[nranks]

    # Discover edge direction dependencies in the mesh locally
    fStart, fEnd = plex.getHeightStratum(1)
    nfacets = fEnd - fStart

    result = np.full(nfacets, -1, dtype=np.int8)
    affects = locally_orient_quadrilateral_plex(plex,
                                                vertex_numbering,
                                                <np.int32_t *>cell_ranks.data,
                                                facet2index,
                                                nfacets_shared,
                                                <np.int8_t *>result.data)
    CHKERR(PetscFree(facet2index))

    # Initialise shared edge directions and assign weights
    #
    # "cabs" of the values in 'ours' and 'theirs' is voting strength, and
    # the sign tells the edge direction. Positive sign implies that the edge
    # points from the vertex with the smaller global number to the vertex with
    # the greater global number, negative implies otherwise.
    ours = MPI.comm.size * np.arange(nfacets_shared, dtype=np.int32) + MPI.comm.rank

    # We update these values based on the local connections
    # before we do any communication.
    for i in range(nfacets_shared):
        if affects[i] != nfacets_shared:
            j = cabs(affects[i])
            if cabs(ours[i]) < cabs(ours[j]):
                if affects[i] >= 0:
                    ours[i] = ours[j]
                else:
                    ours[i] = cneg(ours[j])

    # 'ours' is full of the local view of what the orientations are,
    # and 'theirs' will be filled by the remote orientation view.
    theirs = np.empty_like(ours)

    # Synchronise shared edge directions in parallel
    conflict = int(MPI.comm.size > 1)
    while conflict != 0:
        # Populate 'theirs' by communication from the 'ours' of others.
        exchange_edge_orientation_data(nranks, ranks, offsets, ours, theirs)

        conflict = 0
        for i in range(nfacets_shared):
            if ours[i] != theirs[i] and cabs(ours[i]) == cabs(theirs[i]):
                # Moebius strip found
                raise RuntimeError("Moebius strip found in the mesh.")

            # If the remote value is stronger, ...
            if cabs(ours[i]) < cabs(theirs[i]):
                # ... we adopt it, ...
                ours[i] = theirs[i]

                # ... and propagate, if the other end is shared as well.
                if affects[i] != nfacets_shared:
                    j = cabs(affects[i])  # connected facet at the other end

                    # If the ribbon is twisted locally,
                    # we propagate the orientation accordingly.
                    if affects[i] >= 0:
                        value = ours[i]
                    else:
                        value = cneg(ours[i])

                    # If the other end does not have the same orientation as the
                    # orientation which propagates there, then the twist might
                    # need to travel further in that direction, therefore we
                    # require another round of orientation exchange.
                    if (ours[j] >= 0) ^ (value >= 0):
                        conflict = 1

                    # Please note that at this point cabs(value) is
                    # always greater than cabs(ours[j]).
                    ours[j] = value

        # If there was a conflict anywhere, do another round
        # of communication everywhere.
        conflict = MPI.comm.allreduce(conflict)

    CHKERR(PetscFree(ranks))
    CHKERR(PetscFree(offsets))

    # Reorient the strings of all the shared facets, so that
    # they will match the globally agreed orientations.
    for i in range(nfacets_shared):
        result[facets[i] - fStart] = -1

    for i in range(nfacets_shared):
        f = facets[i]
        if result[f - fStart] == -1:
            if get_global_edge_orientation(plex, vertex_numbering, f) ^ (ours[i] >= 0):
                orientation = 0
            else:
                orientation = 1

            plex_get_restricted_support(plex, <np.int32_t *>cell_ranks.data, f,
                                        &size, cells)

            result[f - fStart] = orientation
            for ci in range(size):
                traverse_cell_string(plex, f, cells[ci],
                                     <np.int32_t *>cell_ranks.data,
                                     <np.int8_t *>result.data)

    CHKERR(PetscFree(facets))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def orientations_facet2cell(
    PETSc.DM plex, PETSc.Section vertex_numbering,
    np.ndarray[np.int32_t, ndim=1, mode="c"] cell_ranks,
    np.ndarray[np.int8_t, ndim=1, mode="c"] facet_orientations,
    PETSc.Section cell_numbering):

    """Converts local quadrilateral facet orientations into
    global quadrilateral cell orientations.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg facet_orientations: Facet orientations (edge directions) relative
                             to the local DMPlex ordering.
    :arg cell_numbering: Section describing the cell numbering
    """
    cdef:
        PetscInt c, cStart, cEnd, ncells, cell
        PetscInt fStart, fEnd
        PetscInt *cone = NULL
        PetscInt *cone_orient = NULL
        np.int8_t dst_orient[4]
        int i, off
        PetscInt facet, v, V
        np.ndarray[np.int32_t, ndim=1, mode="c"] cell_orientations

    cStart, cEnd = plex.getHeightStratum(0)
    fStart, fEnd = plex.getHeightStratum(1)
    ncells = cEnd - cStart

    cell_orientations = np.zeros(ncells, dtype=np.int32)

    for c in range(cStart, cEnd):
        if cell_ranks[c - cStart] < 0:
            CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))

            CHKERR(DMPlexGetCone(plex.dm, c, &cone))
            CHKERR(DMPlexGetConeOrientation(plex.dm, c, &cone_orient))

            # Cone orientations describe which edges to flip (relative to
            # plex edge directions) to get circularly directed edges.
            #
            #   o--<--o
            #   |     |
            #   v     ^
            #   |     |
            #   o-->--o
            #
            # "facet_orientations" describe which edge to flip (relative
            # to plex edge directions) to get edge directions like below
            # for each quadrilateral:
            #
            #   o-->--o
            #   |     |
            #   ^     ^
            #   |     |
            #   X-->--o
            #
            # Their XOR describes the desired edge directions relative to
            # the traversal direction of the cone. This is always a
            # circular permutation of:
            #
            #   straight -- straight -- reverse -- reverse
            #
            for i in range(4):
                dst_orient[i] = (cone_orient[i] < 0) ^ facet_orientations[cone[i] - fStart]

            # We select vertex X (figure above) as starting vertex.
            # Both traversal order (CCW or CW) is fine. We choose the traversal
            # where the second vertex has the smaller global number.
            #
            # The other traversal other would be an equally good choice,
            # however, for cells in the halo, the same choice must be made in
            # each MPI process which sees that cell.
            #
            # To ensure this, we only calculate cell orientations for the
            # locally owned cells, and later exchange these values on the
            # halo cells.
            if dst_orient[2] and dst_orient[3]:
                off = 0
            elif dst_orient[3] and dst_orient[0]:
                off = 1
            elif dst_orient[0] and dst_orient[1]:
                off = 2
            elif dst_orient[1] and dst_orient[2]:
                off = 3
            else:
                raise RuntimeError("Please get the facet orientation right first!")

            # Cell orientation values are defined to be
            # the global number of the starting vertex.
            facet = cone[off]

            CHKERR(DMPlexGetCone(plex.dm, facet, &cone))
            if cone_orient[off] >= 0:
                v = cone[0]
            else:
                v = cone[1]

            CHKERR(PetscSectionGetOffset(vertex_numbering.sec, v, &V))
            cell_orientations[cell] = cabs(V)

    return cell_orientations

@cython.boundscheck(False)
@cython.wraparound(False)
def exchange_cell_orientations(
    PETSc.DM plex, PETSc.Section section,
    np.ndarray[np.int32_t, ndim=1, mode="c"] orientations):

    """Halo exchange of cell orientations.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg section: Section describing the cell numbering
    :arg orientations: Cell orientations to exchange,
                       values in the halo will be overwritten.
    """
    cdef:
        PETSc.SF sf
        PetscInt nroots, nleaves
        PetscInt *ilocal
        PetscSFNode *iremote

        PETSc.Section new_section
        np.int32_t *new_values = NULL
        PetscInt i, c, cStart, cEnd, l, r

    # Halo exchange of cell orientations, i.e. receive orientations
    # from the owners in the halo region.
    if MPI.comm.size > 1:
        sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, &nroots, &nleaves, &ilocal, &iremote))

        new_section = PETSc.Section().create()
        CHKERR(DMPlexDistributeData(plex.dm, sf.sf, section.sec,
                                    MPI_INT, <void *>orientations.data,
                                    new_section.sec, <void **>&new_values))

        # Overwrite values in the halo region with remote values
        cStart, cEnd = plex.getHeightStratum(0)
        for i in range(nleaves):
            c = ilocal[i]
            if cStart <= c < cEnd:
                CHKERR(PetscSectionGetOffset(section.sec, c, &l))
                CHKERR(PetscSectionGetOffset(new_section.sec, c, &r))

                orientations[l] = new_values[r]

    if new_values != NULL:
        CHKERR(PetscFree(new_values))
