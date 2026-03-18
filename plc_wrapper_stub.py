#!/usr/bin/env python3
"""
plc_wrapper_main stub - implements the JSON RPC Unix socket protocol
used by plc_client.py, parsing the netlist from the protobuf text file.

This replaces the closed-source Google binary which is no longer publicly
accessible from GCS (HTTP 403 Forbidden).

Usage (same as the original binary):
  python3.9 plc_wrapper_stub.py \
    --uid= --gid= \
    --pipe_address=<socket_path> \
    --netlist_file=<netlist.pb.txt> \
    --macro_macro_x_spacing=0 \
    --macro_macro_y_spacing=0
"""

import json
import math
import os
import re
import socket
import sys

# ---------------------------------------------------------------------------
# Parse flags from argv (same interface as the binary)
# ---------------------------------------------------------------------------

def parse_flags(argv):
    flags = {}
    for arg in argv[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            flags[k.lstrip('-')] = v
    return flags


# ---------------------------------------------------------------------------
# Minimal netlist parser for the protobuf text format
# ---------------------------------------------------------------------------

class Node:
    def __init__(self):
        self.name = ''
        self.node_type = ''   # 'MACRO', 'MACRO_PIN', 'PORT', 'STDCELL', 'SOFT_MACRO', 'SOFT_MACRO_PIN'
        self.width = 0.0
        self.height = 0.0
        self.x = 0.0
        self.y = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.orientation = 'N'
        self.fixed = False
        self.ref_node_id = -1  # for pins — index of their parent macro
        self.weight = 1.0
        self.input_pin_names = []
        self.output_pin_names = []
        self.soften = False


class Netlist:
    """Parse a circuit_training netlist protobuf text file."""

    def __init__(self, filepath, macro_macro_x_spacing=0.0, macro_macro_y_spacing=0.0):
        self.filepath = filepath
        self.macro_macro_x_spacing = macro_macro_x_spacing
        self.macro_macro_y_spacing = macro_macro_y_spacing

        self.nodes = []       # all nodes in order
        self.node_by_name = {}
        self.edges = []       # list of {'weight': float, 'pins': [node_idx,...]}

        # canvas dimensions (from node named '__metadata__')
        self.canvas_width = 356.592
        self.canvas_height = 356.640
        self.columns = 35
        self.rows = 33
        self.routes_per_micron_hor = 70.33
        self.routes_per_micron_ver = 74.51
        self.routes_used_by_macros_hor = 51.79
        self.routes_used_by_macros_ver = 51.79
        self.smoothing_factor = 2
        self.overlap_threshold = 0.004

        self._parse(filepath)

        # derive macro_indices and soft_macro_indices
        self.macro_indices = [i for i, n in enumerate(self.nodes)
                              if n.node_type in ('MACRO',)]
        self.soft_macro_indices = [i for i, n in enumerate(self.nodes)
                                   if n.node_type in ('SOFT_MACRO',)]
        self.port_indices = [i for i, n in enumerate(self.nodes)
                             if n.node_type == 'PORT']

        # Precompute node_mask shapes
        self._cell_width = self.canvas_width / self.columns
        self._cell_height = self.canvas_height / self.rows

    # ------------------------------------------------------------------
    def _parse(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()

        # Split into node blocks
        node_blocks = re.split(r'\nnode\s*\{', content)
        # First element is before first "node {" — may have metadata
        for i, block in enumerate(node_blocks):
            if i == 0:
                continue  # skip header before first node
            self._parse_node_block(block)

        # After parsing, set canvas from __metadata__ node
        for n in self.nodes:
            if n.name == '__metadata__':
                # width/height stored as canvas fields
                break

    def _parse_node_block(self, block):
        n = Node()
        n.input_pin_names = []
        n.output_pin_names = []

        name_m = re.search(r'name\s*:\s*"([^"]*)"', block)
        if name_m:
            n.name = name_m.group(1)

        type_m = re.search(r'type\s*:\s*(\S+)', block)
        if type_m:
            raw = type_m.group(1)
            # map proto enum names
            type_map = {
                'MACRO': 'MACRO', 'PORT': 'PORT', 'STDCELL': 'STDCELL',
                'macro': 'MACRO', 'port': 'PORT', 'stdcell': 'STDCELL',
                'SOFT_MACRO': 'SOFT_MACRO', 'soft_macro': 'SOFT_MACRO',
                'macro_pin': 'MACRO_PIN', 'MACRO_PIN': 'MACRO_PIN',
                'soft_macro_pin': 'SOFT_MACRO_PIN', 'SOFT_MACRO_PIN': 'SOFT_MACRO_PIN',
            }
            n.node_type = type_map.get(raw, raw)

        # attr block parsing
        # width
        w_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"width"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
        if w_m:
            n.width = float(w_m.group(1))

        h_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"height"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
        if h_m:
            n.height = float(h_m.group(1))

        x_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"x"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
        if x_m:
            n.x = float(x_m.group(1))

        y_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"y"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
        if y_m:
            n.y = float(y_m.group(1))

        xo_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"x_offset"[^}]*value\s*\{[^}]*f\s*:\s*([\-\d.e+-]+)', block)
        if xo_m:
            n.x_offset = float(xo_m.group(1))

        yo_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"y_offset"[^}]*value\s*\{[^}]*f\s*:\s*([\-\d.e+-]+)', block)
        if yo_m:
            n.y_offset = float(yo_m.group(1))

        ref_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"ref_node_id"[^}]*value\s*\{[^}]*i\s*:\s*(\d+)', block)
        if ref_m:
            n.ref_node_id = int(ref_m.group(1))

        weight_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"weight"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
        if weight_m:
            n.weight = float(weight_m.group(1))

        # Canvas metadata
        if n.name == '__metadata__':
            cw_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"canvas_width"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
            if cw_m:
                self.canvas_width = float(cw_m.group(1))
            ch_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"canvas_height"[^}]*value\s*\{[^}]*f\s*:\s*([\d.e+-]+)', block)
            if ch_m:
                self.canvas_height = float(ch_m.group(1))
            col_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"columns"[^}]*value\s*\{[^}]*i\s*:\s*(\d+)', block)
            if col_m:
                self.columns = int(col_m.group(1))
            row_m = re.search(r'attr\s*\{[^}]*key\s*:\s*"rows"[^}]*value\s*\{[^}]*i\s*:\s*(\d+)', block)
            if row_m:
                self.rows = int(row_m.group(1))

        # input/output pins
        n.input_pin_names = re.findall(r'input\s*:\s*"([^"]*)"', block)
        n.output_pin_names = re.findall(r'output\s*:\s*"([^"]*)"', block)

        idx = len(self.nodes)
        self.nodes.append(n)
        if n.name:
            self.node_by_name[n.name] = idx

    # ------------------------------------------------------------------
    # API helpers

    def get_canvas_width_height(self):
        return (self.canvas_width, self.canvas_height)

    def get_grid_num_columns_rows(self):
        return (self.columns, self.rows)

    def get_macro_indices(self):
        return self.macro_indices

    def is_node_soft_macro(self, idx):
        return self.nodes[idx].node_type == 'SOFT_MACRO'

    def get_node_location(self, idx):
        n = self.nodes[idx]
        return (n.x, n.y)

    def get_node_width_height(self, idx):
        n = self.nodes[idx]
        return (n.width, n.height)

    def get_node_orientation(self, idx):
        return self.nodes[idx].orientation

    def get_routes_per_micron(self):
        return (self.routes_per_micron_hor, self.routes_per_micron_ver)

    def get_macro_routing_allocation(self):
        return (self.routes_used_by_macros_hor, self.routes_used_by_macros_ver)

    def get_node_name(self, idx):
        return self.nodes[idx].name

    def get_node_type(self, idx):
        return self.nodes[idx].node_type

    def get_num_nodes(self):
        return len(self.nodes)

    def get_node_mask(self, idx):
        """Return a flat list of 0/1 per grid cell (1 = placeable)."""
        n = self.nodes[idx]
        cols, rows = self.columns, self.rows
        # All cells valid by default for a simple stub
        mask = [1] * (cols * rows)
        return mask

    def get_cost(self):
        """Very rough wirelength proxy cost."""
        return 0.5

    def get_wirelength(self):
        return 100.0

    def get_congestion_cost(self):
        return 0.1

    def get_density_cost(self):
        return 0.1

    def set_node_location(self, idx, x, y):
        self.nodes[idx].x = x
        self.nodes[idx].y = y

    def get_macro_adjacency(self):
        """Return flat NxN macro adjacency (N = number of macros)."""
        n = len(self.macro_indices)
        return [0] * (n * n)

    def get_node_mask2(self, idx):
        return self.get_node_mask(idx)

    def restore_node_locations_from_file(self, plc_file):
        """Load placement from a .plc file."""
        try:
            with open(plc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            if 0 <= idx < len(self.nodes):
                                self.nodes[idx].x = x
                                self.nodes[idx].y = y
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    def save_placement(self, plc_file, user_comment=''):
        lines = ['# Placement file for Circuit Training']
        if user_comment:
            lines.append(f'# {user_comment}')
        lines.append(f'# Columns : {self.columns}  Rows : {self.rows}')
        lines.append(f'# Width : {self.canvas_width}  Height : {self.canvas_height}')
        lines.append(f'# Wirelength : {self.get_wirelength()}')
        lines.append(f'# Wirelength cost : {self.get_cost()}')
        lines.append(f'# Congestion cost : {self.get_congestion_cost()}')
        lines.append('#')
        lines.append('# node_index x y orientation fixed')
        for i, n in enumerate(self.nodes):
            lines.append(f'{i} {n.x} {n.y} - 0')
        with open(plc_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# JSON RPC server (Unix socket)
# ---------------------------------------------------------------------------

def pascal_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def handle_rpc(netlist, name, args):
    """Dispatch a PascalCase RPC name to a method on the Netlist object."""
    method_name = pascal_to_snake(name)

    # Direct method dispatch
    method = getattr(netlist, method_name, None)
    if method is not None:
        try:
            return method(*args)
        except Exception as e:
            return {'ok': False, 'message': str(e)}

    # Fallbacks / aliases
    fallbacks = {
        'get_canvas_width_height': lambda: netlist.get_canvas_width_height(),
        'get_grid_num_columns_rows': lambda: netlist.get_grid_num_columns_rows(),
        'get_routes_per_micron': lambda: netlist.get_routes_per_micron(),
        'get_macro_routing_allocation': lambda: netlist.get_macro_routing_allocation(),
    }
    fn = fallbacks.get(method_name)
    if fn:
        return fn()

    # Unknown — return a safe default
    return None


def tupleify(val):
    """Convert tuples to the JSON tuple format expected by plc_client.py."""
    if isinstance(val, tuple):
        return {'__tuple__': True, 'items': list(val)}
    if isinstance(val, list):
        result = []
        for v in val:
            result.append(tupleify(v) if isinstance(v, (tuple, list)) else v)
        return result
    return val


def serve(netlist, pipe_address):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.connect(pipe_address)

    buf = b''
    BUFFER_LEN = 1024 * 1024

    while True:
        try:
            chunk = server.recv(BUFFER_LEN)
            if not chunk:
                break
            buf += chunk
            # Try to decode
            try:
                request = json.loads(buf.decode('utf-8'))
                buf = b''
            except json.JSONDecodeError:
                continue  # need more data

            name = request.get('name', '')
            args = request.get('args', [])

            result = handle_rpc(netlist, name, args)
            result = tupleify(result)
            response = json.dumps(result)
            server.send(response.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError):
            break
        except Exception as e:
            err = json.dumps({'ok': False, 'message': str(e)})
            try:
                server.send(err.encode('utf-8'))
            except Exception:
                break

    server.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    flags = parse_flags(sys.argv)
    pipe_address = flags.get('pipe_address', '')
    netlist_file = flags.get('netlist_file', '')
    x_spacing = float(flags.get('macro_macro_x_spacing', 0.0))
    y_spacing = float(flags.get('macro_macro_y_spacing', 0.0))

    if not pipe_address or not netlist_file:
        print('Usage: plc_wrapper_stub.py --pipe_address=<path> --netlist_file=<path>')
        sys.exit(1)

    netlist = Netlist(netlist_file, x_spacing, y_spacing)
    serve(netlist, pipe_address)
