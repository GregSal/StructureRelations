'''Build Relation Diagram.
'''
# %% Imports
# Type imports

from typing import List, Tuple, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Standard Libraries
from itertools import chain
from math import ceil, sin, cos, radians, sqrt
from collections import defaultdict


# Shared Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
import pydicom
from shapely.plotting import plot_polygon, plot_line
import pygraphviz as pgv

from types_and_classes import ROI_Num, SliceIndex, Contour, StructurePair, poly_round
from types_and_classes import InvalidContour

from types_and_classes import StructureSlice, RelationshipType
from relations import Relationship
from structure import Structure


# Global Default Settings
PRECISION = 3


# %% StructureDiagram class
class StructureDiagram:
    graph_defaults = {
        'labelloc': 't',
        'clusterrank': 'none',
        'bgcolor': '#555555',
        'fontname': 'Helvetica,,Arial,sans-serif',
        'fontsize': 16,
        'fontcolor': 'white'
        }
    node_defaults = {
        'style': 'filled',
        'width': 1,
        'height': 0.6,
        'fixedsize': 'shape',
        'fontname': 'Helvetica-Bold',
        'fontsize': 12,
        'fontcolor': 'black',
        'labelloc': 'c',
        'nojustify': True,
        'penwidth': 3,
        }
    edge_defaults = {
        'style': 'solid',
        'penwidth': 3,
        'color': '#e27dd6ff',
        'arrowhead': 'none',
        'arrowtail': 'none',
        'labelfloat': False,
        'labelfontname': 'Cambria',
        'fontsize': '10',
        'fontcolor': '#55AAFF',
        }
    node_type_formatting = {
        'GTV': {'shape': 'pentagon', 'style': 'filled', 'penwidth': 3},
        'CTV': {'shape': 'hexagon', 'style': 'filled', 'penwidth': 3},
        'PTV': {'shape': 'octagon', 'style': 'filled', 'penwidth': 3},
        'EXTERNAL': {'shape': 'doublecircle', 'style': 'filled',
                     'fillcolor': 'white','penwidth': 2},
        'ORGAN': {'shape': 'rectangle', 'style': 'rounded, filled',
                  'penwidth': 3},
        'NONE': {'shape': 'trapezium', 'style': 'rounded, filled',
                 'penwidth': 3},
        'AVOIDANCE': {'shape': 'house', 'style': 'rounded, filled',
                 'penwidth': 3},
        'CONTROL': {'shape': 'invhouse', 'style': 'rounded, filled',
                 'penwidth': 3},
        'TREATED_VOLUME': {'shape': 'parallelogram', 'style': 'rounded, filled',
                 'penwidth': 3},
        'IRRAD_VOLUME': {'shape': 'parallelogram', 'style': 'rounded, filled',
                 'penwidth': 3},
        'DOSE_REGION': {'shape': 'diamond', 'style': 'rounded, filled',
                 'penwidth': 3},
        'CONTRAST_AGENT': {'shape': 'square', 'style': 'rounded, filled',
                 'penwidth': 3},
        'CAVITY': {'shape': 'square', 'style': 'rounded, filled',
                 'penwidth': 3},
        'SUPPORT': {'shape': 'triangle', 'style': 'rounded, bold',
                 'penwidth': 3},
        'BOLUS': {'shape': 'oval', 'style': 'bold', 'penwidth': 3},
        'FIXATION': {'shape': 'diamond', 'style': 'bold', 'penwidth': 3},
        }
    edge_type_formatting = {
        RelationshipType.DISJOINT: {'label': 'Disjoint', 'style': 'invis'},
        RelationshipType.SURROUNDS: {'label': 'Island', 'style': 'tapered',
                                     'dir': 'forward', 'penwidth': 3,
                                     'color': 'blue'},
        RelationshipType.SHELTERS: {'label': 'Shelter', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 3, 'color': 'blue'},
        RelationshipType.BORDERS: {'label': 'Borders', 'style': 'dashed',
                                   'dir': 'both', 'penwidth': 3,
                                   'color': 'green'},
        RelationshipType.BORDERS_INTERIOR: {'label': 'Cut-out', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 3,
                                    'color': 'magenta'},
        RelationshipType.OVERLAPS: {'label': 'Overlaps', 'style': 'tapered',
                                    'dir': 'both', 'penwidth': 6,
                                    'color': 'green'},
        RelationshipType.PARTITION: {'label': 'Group', 'style': 'tapered',
                                        'dir': 'forward', 'penwidth': 6,
                                        'color': 'white'},
        RelationshipType.CONTAINS: {'label': 'Contains', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 6,
                                    'color': 'cyan'},
        RelationshipType.EQUALS: {'label': 'Equals', 'style': 'both',
                                  'dir': 'both', 'penwidth': 5, 'color': 'red'},
        RelationshipType.LOGICAL: {'label': '', 'style': 'dotted',
                                   'penwidth': 0.5, 'color': 'yellow'},
        RelationshipType.UNKNOWN: {'label': '', 'style': 'invis'},
        }

    # The Formatting style for hidden structures and relationships
    hidden_node_format = {'shape': 'point', 'style': 'invis'}
    hidden_edge_format = {'style': 'invis'}
    logical_edge_format = {'style': 'dotted', 'penwidth': 0.5,
                           'color': 'yellow'}

    def __init__(self, name=r'Structure Relations') -> None:
        self.title = name
        self.display_graph = pgv.AGraph(label=name, **self.graph_defaults)
        self.display_graph.node_attr.update(self.node_defaults)
        self.display_graph.edge_attr.update(self.edge_defaults)

    @staticmethod
    def rgb_to_hex(rgb_tuple: Tuple[int,int,int]) -> str:
        '''Convert an RGB tuple to a hex string value.

        Args:
            rgb_tuple (Tuple[int,int,int]): A length-3 tuple of integers from 0 to
                255 corresponding to the RED, GREEN and BLUE color channels.

        Returns:
            str: The equivalent Hex color string.
        '''
        return '#{:02x}{:02x}{:02x}'.format(*rgb_tuple)

    @staticmethod
    def text_color(color_rgb: Tuple[int])->Tuple[int]:
        '''Determine the appropriate text color for a given background color.

        Text color is either Black (0, 0, 0) or white (255, 255, 255)
        the cutoff between black and white is given by:
            brightness > 274.3 and green > 69
            (brightness is the length of the color vector $sqrt{R^2+G^2+B62}$

        Args:
            color_rgb (Tuple[int]): The 3-integer RGB tuple of the background color.

        Returns:
            Tuple[int]: The text color as an RGB tuple.
                One of (0, 0, 0) or (255, 255, 255).
        '''
        red, green, blue = color_rgb
        brightness = sqrt(red**2 + green**2 + blue**2)
        if brightness > 274.3:
            if green > 69:
                text_color = '#000000'
            else:
                text_color = '#FFFFFF'
        elif green > 181:
            text_color = '#000000'
        else:
            text_color = '#FFFFFF'
        return text_color

    def add_structure_nodes(self, structures: List[Structure]):
        structure_groups = defaultdict(list)
        for structure in structures:
            node_type = structure.structure_type
            node_id = structure.roi_num
            node_formatting = self.node_type_formatting[node_type].copy()
            node_text_color = self.text_color(structure.color)
            node_formatting['label'] = structure.info.structure_id
            node_formatting['color'] = self.rgb_to_hex(structure.color)
            node_formatting['fontcolor'] = node_text_color
            node_formatting['tooltip'] = structure.summary()
            if not structure.show:
                node_formatting.update(self.hidden_node_format)
            self.display_graph.add_node(node_id, **node_formatting)
            # Identify the subgroup that the node belongs to.
            group = structure.structure_category
            structure_groups[group].append(structure.roi_num)
        # Define the subgroups.
        for name, group_list in structure_groups.items():
            self.display_graph.add_subgraph(group_list, name=str(name),
                                            cluster=True)

    def add_structure_edges(self, relationships: List[Relationship]):
        for relationship in relationships:
            node1, node2 = relationship.structures
            edge_type = relationship.relationship_type
            edge_formatting = self.edge_type_formatting[edge_type].copy()
            edge_formatting['tooltip'] = relationship.metric.format_metric()
            # Override formatting
            hide_edge = any(self.is_hidden(node)
                            for node in relationship.structures)
            if (not relationship.show) | hide_edge:
                edge_formatting.update(self.hidden_edge_format)
            elif relationship.is_logical:
                edge_formatting.update(self.logical_edge_format)
            self.display_graph.add_edge(node1, node2, **edge_formatting)

    def node_attr(self, node_id: int)->Dict[str, any]:
        return self.display_graph.get_node(node_id).attr

    def is_hidden(self, node_id: int)->bool:
        node_attributes = self.node_attr(node_id)
        node_style = node_attributes.get('style', '')
        return 'invis' in node_style

# %% Debugging Display functions
# Eventually move these functions to their own module
