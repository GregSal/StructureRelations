import pytest
import shapely
import numpy as np
import networkx as nx
from contour_graph import build_contour_graph
from types_and_classes import RegionIndex, SliceIndexType
from region_slice import RegionSlice, empty_structure
from utilities import make_multi
from debug_tools import box_points
from contours import ContourPoints, build_contour_table

from region_slice import RegionSlice

class TestRegionSlice:
    '''Test cases for the RegionSlice class'''
    def test_region_slice_with_one_contour(self):
        box1 = box_points(width=2)
        slice_data = [
            ContourPoints(box1, roi=1, slice_index=1.0),  # ROI 1, Area 4
            ContourPoints(box1, roi=1, slice_index=2.0),  # ROI 1, Area 4
            ContourPoints(box1, roi=1, slice_index=3.0),  # ROI 1, Area 4
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        region_slice = RegionSlice(contour_graph, slice_index=1.0)
        # --- Begin assertions for the specified properties ---
        # There should be a single region
        assert len(region_slice.regions) == 1
        # The region should be a MultiPolygon and not empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert not region.is_empty
        # Boundaries should be an empty MultiPolygon
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert boundary.is_empty
        # Open holes should be an empty MultiPolygon
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Exterior should be the same as the region
        # pylint: disable=consider-using-dict-items
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            exterior = region_slice.exterior[region_index]
            assert exterior.equals(region)
        # Hull should be the same as the region
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            hull = region_slice.hull[region_index]
            assert hull.equals(region)
        # Embedded regions should be an empty list
        for embedded in region_slice.embedded_regions.values():
            assert embedded == []
        # Region holes should be an empty list
        for holes in region_slice.region_holes.values():
            assert holes == []
        # Contour indexes should contain a single ContourIndex
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 1

    def test_region_slice_two_nonoverlapping(self):
        # Create two non-overlapping boxes on the same slice
        box1 = box_points(width=2, offset_x=-3)
        box2 = box_points(width=2, offset_x=3)
        slice_data = [
            ContourPoints(box1, roi=1, slice_index=1.0),
            ContourPoints(box2, roi=1, slice_index=1.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        region_slice = RegionSlice(contour_graph, slice_index=1.0)
        # There should be two regions
        assert len(region_slice.regions) == 2
        # Each region should be a MultiPolygon and not empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert not region.is_empty
        # Boundaries should be two empty MultiPolygons
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert boundary.is_empty
        # Open holes should be two empty MultiPolygons
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Exterior should be the same as the region
        # pylint: disable=consider-using-dict-items
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            exterior = region_slice.exterior[region_index]
            assert exterior.equals(region)
        # Hull should be the same as the region
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            hull = region_slice.hull[region_index]
            assert hull.equals(region)
        # Embedded regions should be empty lists
        for embedded in region_slice.embedded_regions.values():
            assert embedded == []
        # Region holes should be empty lists
        for holes in region_slice.region_holes.values():
            assert holes == []
        # Contour indexes should contain one ContourIndex each (total two)
        index_count  = [len(indexes)
                        for indexes in region_slice.contour_indexes.values()]
        total_indexes = sum(indexes for indexes in index_count)
        assert total_indexes == 2

    def test_region_slice_two_nonoverlapping_same_region(self):
        # Create two non-overlapping boxes on the same slice, force same region index
        box1 = box_points(width=2, offset_x=-3)
        box2 = box_points(width=2, offset_x=3)
        slice_data = [
            ContourPoints(box1, roi=1, slice_index=1.0),
            ContourPoints(box2, roi=1, slice_index=1.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        # Force both contours to have the same region index
        region_indexes = set()
        for node in contour_graph.nodes:
            region_indexes.add(contour_graph.nodes[node]['contour'].region_index)
        # If more than one region, set all to the same region index
        if len(region_indexes) > 1:
            first_region = next(iter(region_indexes))
            for node in contour_graph.nodes:
                contour_graph.nodes[node]['contour'].region_index = first_region

        region_slice = RegionSlice(contour_graph, slice_index=1.0)
        # There should be a single region
        assert len(region_slice.regions) == 1
        # The region should be a MultiPolygon and not empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert not region.is_empty
            # Should have two polygons inside the multipolygon
            assert len(region.geoms) == 2
        # Boundaries should be an empty MultiPolygon
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert boundary.is_empty
        # Open holes should be an empty MultiPolygon
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Exterior should be the same as the region
        # pylint: disable=consider-using-dict-items
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            exterior = region_slice.exterior[region_index]
            assert exterior.equals(region)
        # Hull is the convex hull of the region MultiPolygon
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            hull = region_slice.hull[region_index]
            assert hull.equals(region.convex_hull)
        # Embedded regions should be empty lists
        for embedded in region_slice.embedded_regions.values():
            assert embedded == []
        # Region holes should be empty lists
        for holes in region_slice.region_holes.values():
            assert holes == []
        # Contour indexes should contain two ContourIndexes
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 2

    def test_region_slice_with_open_hole(self):
        # Create a large box and a smaller box (hole) inside, both on the same slice
        outer = box_points(width=4)
        hole = box_points(width=2)
        slice_data = [
            ContourPoints(outer, roi=1, slice_index=1.0),
            ContourPoints(hole, roi=1, slice_index=1.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        # Force the hole contour to be marked as an open hole
        # (simulate what set_hole_type would do in a real open hole scenario)
        # Find the smaller contour and set its hole_type to 'Open'
        for node in contour_graph.nodes:
            contour = contour_graph.nodes[node]['contour']
            if contour.polygon.area < 10:  # area of the hole
                contour.is_hole = True
                contour.hole_type = 'Open'

        region_slice = RegionSlice(contour_graph, slice_index=1.0)
        # Regions should contain two MultiPolygons, one of which is empty.
        assert len(region_slice.regions) == 2
        non_empty_regions = [region for region in region_slice.regions.values()
                             if not region.is_empty]
        assert len(non_empty_regions) == 1
        # The region should be a MultiPolygon and not empty
        assert isinstance(non_empty_regions[0], shapely.MultiPolygon)
        assert not non_empty_regions[0].is_empty
        # Boundaries should contain two empty MultiPolygons
        assert len(region_slice.boundaries) == 2
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert boundary.is_empty
        # Open holes should contain the hole as a MultiPolygon plus an empty
        #  MultiPolygon
        open_holes = list(region_slice.open_holes.values())
        assert len(open_holes) == 2
        # One should be non-empty and match the hole, the other should be empty
        non_empty = [oh for oh in open_holes if not oh.is_empty]
        empty = [oh for oh in open_holes if oh.is_empty]
        assert len(non_empty) == 1
        assert len(empty) == 1
        assert isinstance(non_empty[0], shapely.MultiPolygon)
        assert non_empty[0].geoms[0].area == pytest.approx(shapely.Polygon(hole).area)
        # Exterior should be the same as the region
        # pylint: disable=consider-using-dict-items
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            exterior = region_slice.exterior[region_index]
            assert exterior.equals(region)
        # Hull contains the larger contour as a single MultiPolygon plus an
        # empty MultiPolygon
        hulls = list(region_slice.hull.values())
        assert len(hulls) == 2
        non_empty_hulls = [h for h in hulls if not h.is_empty]
        empty_hulls = [h for h in hulls if h.is_empty]
        assert len(non_empty_hulls) == 1
        assert len(empty_hulls) == 1
        # The non-empty hull should be the convex hull of the non-empty region
        non_empty_regions = [region for region in region_slice.regions.values()
                             if not region.is_empty]
        assert non_empty_hulls[0].equals(non_empty_regions[0].convex_hull)
        # Embedded regions should contain the hole as a single Contour and an
        # empty list
        embedded_lists = list(region_slice.embedded_regions.values())
        assert len(embedded_lists) == 2
        non_empty_embedded = [emb for emb in embedded_lists if len(emb) > 0]
        empty_embedded = [emb for emb in embedded_lists if len(emb) == 0]
        assert len(non_empty_embedded) == 1
        assert len(empty_embedded) == 1
        assert non_empty_embedded[0][0].polygon.area == pytest.approx(shapely.Polygon(hole).area)
        # Region holes should contain the hole as a single Contour and an
        # empty list
        region_holes_lists = list(region_slice.region_holes.values())
        assert len(region_holes_lists) == 2
        non_empty_holes = [holes for holes in region_holes_lists if len(holes) == 1]
        empty_holes = [holes for holes in region_holes_lists if len(holes) == 0]
        assert len(non_empty_holes) == 1
        assert len(empty_holes) == 1
        assert non_empty_holes[0][0].polygon.area == pytest.approx(shapely.Polygon(hole).area)
        # contour_indexes should contain two items, each a list containing a
        # single ContourIndex
        assert len(region_slice.contour_indexes) == 2
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 1

    def test_region_slice_with_closed_hole(self):
        # Create a large box and a smaller box (hole) inside, both on the same slice
        outer = box_points(width=4)
        hole = box_points(width=2)
        slice_data = [
            ContourPoints(outer, roi=1, slice_index=1.0),
            ContourPoints(hole, roi=1, slice_index=1.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        # Force the hole contour to be marked as a closed hole
        for node in contour_graph.nodes:
            contour = contour_graph.nodes[node]['contour']
            if contour.polygon.area < 10:
                contour.is_hole = True
                contour.hole_type = 'Closed'

        region_slice = RegionSlice(contour_graph, slice_index=1.0)
        # There should be a single region
        assert len(region_slice.regions) == 1
        # The region should be a MultiPolygon and not empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert not region.is_empty
        # Boundaries should be an empty MultiPolygon
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert boundary.is_empty
        # Open holes should be an empty MultiPolygon
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Exterior contains the larger contour as a MultiPolygon
        # pylint: disable=consider-using-dict-items
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            exterior = region_slice.exterior[region_index]
            # The exterior should fill the closed hole, so area equals outer box
            assert exterior.area == pytest.approx(shapely.Polygon(outer).area)
        # Hull is the convex hull of the region MultiPolygon
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            hull = region_slice.hull[region_index]
            assert hull.equals(region.convex_hull)
        # Embedded regions should be empty lists
        for embedded in region_slice.embedded_regions.values():
            assert embedded == []
        # Region holes should contain a single Contour
        for holes in region_slice.region_holes.values():
            assert len(holes) == 1
        # Contour indexes should contain two ContourIndexes
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 2

    def test_region_slice_with_closed_hole_and_island(self):
        # Create a large box, a smaller box (hole) inside, and an island inside the hole, all on the same slice
        outer = box_points(width=4)
        hole = box_points(width=2)
        island = box_points(width=1)
        slice_data = [
            ContourPoints(outer, roi=1, slice_index=1.0),
            ContourPoints(hole, roi=1, slice_index=1.0),
            ContourPoints(island, roi=1, slice_index=1.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        # Force the hole contour to be marked as a closed hole
        # and the island to be a non-hole inside the hole
        hole_contour = None
        island_contour = None
        for node in contour_graph.nodes:
            contour = contour_graph.nodes[node]['contour']
            if contour.polygon.area == pytest.approx(shapely.Polygon(hole).area):
                contour.is_hole = True
                contour.hole_type = 'Closed'
                hole_contour = contour
            elif contour.polygon.area == pytest.approx(shapely.Polygon(island).area):
                # This is the island, ensure it's not a hole
                contour.is_hole = False
                island_contour = contour

        region_slice = RegionSlice(contour_graph, slice_index=1.0)
        # There should be a single region
        assert len(region_slice.regions) == 1
        # The region should be a MultiPolygon and not empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert not region.is_empty
        # Boundaries should be an empty MultiPolygon
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert boundary.is_empty
        # Open holes should be an empty MultiPolygon
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Exterior contains the larger contour as a MultiPolygon
        # pylint: disable=consider-using-dict-items
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            exterior = region_slice.exterior[region_index]
            assert exterior.area == pytest.approx(shapely.Polygon(outer).area)
        # Hull is the convex hull of the region MultiPolygon
        for region_index in region_slice.regions:
            region = region_slice.regions[region_index]
            hull = region_slice.hull[region_index]
            assert hull.equals(region.convex_hull)
        # Embedded regions contains a single Contour (the island)
        for embedded in region_slice.embedded_regions.values():
            assert len(embedded) == 1
            assert embedded[0].polygon.area == pytest.approx(shapely.Polygon(island).area)
        # Region holes should contain a single Contour (the hole)
        for holes in region_slice.region_holes.values():
            assert len(holes) == 1
            assert holes[0].polygon.area == pytest.approx(shapely.Polygon(hole).area)
        # Contour indexes should contain three ContourIndexes
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 3

    def test_boundary_slice_with_one_contour_per_slice(self):
        # Create a box and build the contour graph
        box1 = box_points(width=2)
        slice_data = [
            ContourPoints(box1, roi=1, slice_index=1.0),
            ContourPoints(box1, roi=1, slice_index=2.0),
            ContourPoints(box1, roi=1, slice_index=3.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        # Find a boundary slice (should be at the interpolated boundary, e.g., 0.5 or 3.5)
        # Get all interpolated slices from the sequence
        not_original = slice_sequence.sequence.Original == False
        interpolated_slice_indexes = list(slice_sequence.sequence.loc[not_original, 'ThisSlice'])
        # Pick the first interpolated slice
        boundary_slice_index = interpolated_slice_indexes[0]
        region_slice = RegionSlice(contour_graph, slice_index=boundary_slice_index)
        # There should be a single boundary
        assert len(region_slice.boundaries) == 1
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert not boundary.is_empty
        # Regions should be empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert region.is_empty
        # Open holes should be empty
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Embedded regions should be empty lists
        for embedded in region_slice.embedded_regions.values():
            assert embedded == []
        # Region holes should be empty lists
        for holes in region_slice.region_holes.values():
            assert holes == []
        # Contour indexes should contain a single ContourIndex
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 1
        # is_interpolated should be True
        assert region_slice.is_interpolated is True

    def test_boundary_slice_for_open_hole(self):
        # Create a large box and a smaller box (hole) inside, both on the same slice
        outer = box_points(width=4)
        hole = box_points(width=2)
        slice_data = [
            ContourPoints(outer, roi=1, slice_index=1.0),
            ContourPoints(hole, roi=1, slice_index=1.0),
            ContourPoints(outer, roi=1, slice_index=2.0),
            ContourPoints(hole, roi=1, slice_index=2.0),
            ContourPoints(outer, roi=1, slice_index=3.0),
            ContourPoints(hole, roi=1, slice_index=3.0),
        ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            roi=1)
        # Force the hole contour to be marked as an open hole
        for node in contour_graph.nodes:
            contour = contour_graph.nodes[node]['contour']
            if contour.polygon.area < 10:
                contour.is_hole = True
                contour.hole_type = 'Open'

        # Find a boundary slice (should be at the interpolated boundary, e.g., 0.5 or 1.5)
        not_original = slice_sequence.sequence.Original == False
        interpolated_slice_indexes = list(slice_sequence.sequence.loc[not_original, 'ThisSlice'])
        boundary_slice_index = interpolated_slice_indexes[0]
        region_slice = RegionSlice(contour_graph, slice_index=boundary_slice_index)
        # There should be a single boundary
        assert len(region_slice.boundaries) == 1
        for boundary in region_slice.boundaries.values():
            assert isinstance(boundary, shapely.MultiPolygon)
            assert not boundary.is_empty
        # Regions should be empty
        for region in region_slice.regions.values():
            assert isinstance(region, shapely.MultiPolygon)
            assert region.is_empty
        # Open holes should be empty
        for open_hole in region_slice.open_holes.values():
            assert isinstance(open_hole, shapely.MultiPolygon)
            assert open_hole.is_empty
        # Embedded regions should be empty lists
        for embedded in region_slice.embedded_regions.values():
            assert embedded == []
        # Region holes should contain a single Contour
        for holes in region_slice.region_holes.values():
            assert len(holes) == 1
        # Contour indexes should contain a single ContourIndex
        for indexes in region_slice.contour_indexes.values():
            assert len(indexes) == 1
        # is_interpolated should be True
        assert region_slice.is_interpolated is True
