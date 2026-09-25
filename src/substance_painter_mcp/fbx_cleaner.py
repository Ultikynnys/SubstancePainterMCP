"""
FBX Cleaner Engine for Substance 3D Painter.
Pure Python FBX parser and processor (supports both Binary and ASCII FBX).
Filters out UCX / collision meshes and LODs (preserving LOD0 / base meshes).
"""

import io
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# FBX Binary Magic & Constants
BINARY_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

COLLIDER_REGEXES = [
    re.compile(r'^(ucx|ubx|usp|ucl)[_\-]', re.IGNORECASE),
    re.compile(r'[_\-](ucx|ubx|usp|ucl)[_\-]', re.IGNORECASE),
    re.compile(r'[_\-](ucx|ubx|usp|ucl)$', re.IGNORECASE),
    re.compile(r'^(col|mcol|collider|collision)[_\-]', re.IGNORECASE),
    re.compile(r'[_\-](col|mcol|collider|collision)$', re.IGNORECASE),
    re.compile(r'[_\-](col|mcol|collider|collision)[_\-]', re.IGNORECASE),
]

LOD_REGEXES = [
    re.compile(r'[_\.\-\s]lod(\d+)$', re.IGNORECASE),
    re.compile(r'[_\.\-\s]lod(\d+)[_\.\-\s]', re.IGNORECASE),
    re.compile(r'^lod(\d+)[_\.\-\s]', re.IGNORECASE),
]


def clean_node_name(raw_name: str) -> str:
    """Strip FBX class prefix/suffix from node names."""
    if '\x00\x01' in raw_name:
        raw_name = raw_name.split('\x00\x01')[0]
    if '::' in raw_name:
        parts = raw_name.split('::')
        if parts[0] in ('Model', 'Geometry', 'NodeAttribute', 'Material'):
            raw_name = parts[1]
        else:
            raw_name = parts[0]
    return raw_name


@dataclass
class MeshInfo:
    id: int
    raw_name: str
    clean_name: str
    classification: str  # "COLLIDER", "LOD", "BASE_LOD0", "BASE_MESH"
    lod_index: Optional[int] = None
    default_discard: bool = False
    reason: str = ""

    @property
    def should_discard(self) -> bool:
        return self.default_discard


@dataclass
class CleanOptions:
    discard_colliders: bool = True
    discard_lods: bool = True
    keep_lod0: bool = True
    custom_filter_regex: Optional[str] = None


@dataclass
class CleanReport:
    input_path: str
    output_path: str
    format: str  # "binary" or "ascii"
    total_meshes: int = 0
    kept_mesh_names: List[str] = field(default_factory=list)
    discarded_collider_names: List[str] = field(default_factory=list)
    discarded_lod_names: List[str] = field(default_factory=list)
    discarded_custom_names: List[str] = field(default_factory=list)

    @property
    def kept_meshes(self) -> List[str]:
        return self.kept_mesh_names

    @property
    def discarded_colliders(self) -> List[str]:
        return self.discarded_collider_names

    @property
    def discarded_lods(self) -> List[str]:
        return self.discarded_lod_names


def classify_mesh(name: str, options: CleanOptions) -> Tuple[str, Optional[int], bool, str]:
    """Classify mesh as COLLIDER, LOD, BASE_LOD0, or BASE_MESH."""
    clean = clean_node_name(name)

    # Check custom regex if provided
    if options.custom_filter_regex:
        try:
            if re.search(options.custom_filter_regex, clean, re.IGNORECASE):
                return "CUSTOM", None, True, "Matches custom filter"
        except re.error:
            pass

    # Check collider patterns
    for pat in COLLIDER_REGEXES:
        if pat.search(clean):
            return "COLLIDER", None, options.discard_colliders, "UCX / Collider Mesh"

    # Check LOD patterns
    for pat in LOD_REGEXES:
        m = pat.search(clean)
        if m:
            lod_idx = int(m.group(1))
            if lod_idx == 0:
                return "BASE_LOD0", 0, False, "LOD 0 (Base Detail Mesh)"
            else:
                return "LOD", lod_idx, options.discard_lods, f"LOD Level {lod_idx}"

    return "BASE_MESH", None, False, "Base Mesh"


# ==============================================================================
# Binary FBX Representation
# ==============================================================================

class FBXProperty:
    __slots__ = ('type_code', 'raw_bytes', 'value')

    def __init__(self, type_code: str, raw_bytes: bytes, value=None):
        self.type_code = type_code
        self.raw_bytes = raw_bytes
        self.value = value

    @classmethod
    def from_int32(cls, val: int):
        return cls('I', b'I' + struct.pack('<i', val), val)

    @classmethod
    def from_int64(cls, val: int):
        return cls('L', b'L' + struct.pack('<q', val), val)

    @classmethod
    def from_string(cls, val: str):
        b = val.encode('utf-8')
        return cls('S', b'S' + struct.pack('<I', len(b)) + b, val)

    def __repr__(self):
        return f"Prop({self.type_code}, {self.value!r})"


class FBXNode:
    __slots__ = ('name', 'properties', 'children', 'has_sentinel')

    def __init__(self, name="", properties=None, children=None, has_sentinel=False):
        self.name = name.encode('utf-8') if isinstance(name, str) else name
        self.properties = properties or []
        self.children = children or []
        self.has_sentinel = has_sentinel

    @property
    def name_str(self) -> str:
        return self.name.decode('utf-8', errors='replace')

    @property
    def raw_properties_data(self) -> bytes:
        return b"".join(p.raw_bytes for p in self.properties)

    def get_child(self, name: str) -> Optional['FBXNode']:
        target = name.encode('utf-8') if isinstance(name, str) else name
        for c in self.children:
            if c.name == target:
                return c
        return None

    def get_children(self, name: str) -> List['FBXNode']:
        target = name.encode('utf-8') if isinstance(name, str) else name
        return [c for c in self.children if c.name == target]

    def __repr__(self):
        return f"Node({self.name_str}, props={len(self.properties)}, children={len(self.children)})"


def _read_binary_property(stream) -> FBXProperty:
    tc_bytes = stream.read(1)
    if not tc_bytes:
        raise EOFError("Unexpected EOF while reading property type")
    tc = tc_bytes[0]

    if tc == ord('Y'):
        data = stream.read(2)
        val = struct.unpack('<h', data)[0]
    elif tc in (ord('C'), ord('B')):
        data = stream.read(1)
        val = bool(struct.unpack('<?', data)[0])
    elif tc == ord('I'):
        data = stream.read(4)
        val = struct.unpack('<i', data)[0]
    elif tc == ord('F'):
        data = stream.read(4)
        val = struct.unpack('<f', data)[0]
    elif tc == ord('D'):
        data = stream.read(8)
        val = struct.unpack('<d', data)[0]
    elif tc == ord('L'):
        data = stream.read(8)
        val = struct.unpack('<q', data)[0]
    elif tc == ord('Z'):
        data = stream.read(1)
        val = struct.unpack('<b', data)[0]
    elif tc in (ord('S'), ord('R')):
        len_data = stream.read(4)
        length = struct.unpack('<I', len_data)[0]
        payload = stream.read(length)
        data = len_data + payload
        val = payload.decode('utf-8', errors='replace') if tc == ord('S') else payload
    elif tc in (ord('b'), ord('c'), ord('i'), ord('l'), ord('f'), ord('d')):
        header_data = stream.read(12)
        array_len, encoding, comp_len = struct.unpack('<III', header_data)
        if encoding == 1:
            payload = stream.read(comp_len)
        else:
            stride = {ord('b'): 1, ord('c'): 1, ord('i'): 4, ord('l'): 8, ord('f'): 4, ord('d'): 8}[tc]
            payload = stream.read(array_len * stride)
        data = header_data + payload
        val = f"<array {chr(tc)} len={array_len}>"
    else:
        raise ValueError(f"Unknown property type code: {tc}")

    raw_bytes = tc_bytes + data
    return FBXProperty(type_code=chr(tc), raw_bytes=raw_bytes, value=val)


def _read_binary_node(stream, is_64bit: bool) -> Optional[FBXNode]:
    header_len = 25 if is_64bit else 13
    header = stream.read(header_len)
    if len(header) < header_len:
        return None

    if is_64bit:
        end_offset, num_props, prop_len, name_len = struct.unpack('<QQQB', header)
    else:
        end_offset, num_props, prop_len, name_len = struct.unpack('<IIIB', header)

    if end_offset == 0 and num_props == 0 and prop_len == 0 and name_len == 0:
        return None

    name = stream.read(name_len)
    props = []
    prop_start = stream.tell()
    for _ in range(num_props):
        props.append(_read_binary_property(stream))

    bytes_read = stream.tell() - prop_start
    if bytes_read < prop_len:
        stream.read(prop_len - bytes_read)

    children = []
    has_sentinel = False
    curr_pos = stream.tell()
    if curr_pos < end_offset:
        has_sentinel = True
        while stream.tell() < end_offset:
            child = _read_binary_node(stream, is_64bit)
            if child is None:
                break
            children.append(child)

        if stream.tell() < end_offset:
            stream.seek(end_offset)

    return FBXNode(name=name, properties=props, children=children, has_sentinel=has_sentinel)


def parse_binary_fbx_stream(stream) -> Tuple[int, List[FBXNode], bytes]:
    magic = stream.read(len(BINARY_MAGIC))
    if magic != BINARY_MAGIC:
        raise ValueError("Not a valid binary FBX file")
    version = struct.unpack('<I', stream.read(4))[0]
    is_64bit = version >= 7500

    root_nodes = []
    while True:
        node = _read_binary_node(stream, is_64bit)
        if node is None:
            break
        root_nodes.append(node)

    footer = stream.read()
    return version, root_nodes, footer


def _compute_binary_node_size(node: FBXNode, is_64bit: bool) -> int:
    header_size = 25 if is_64bit else 13
    name_len = len(node.name)
    prop_len = len(node.raw_properties_data)
    size = header_size + name_len + prop_len

    has_subscope = len(node.children) > 0 or node.has_sentinel
    if has_subscope:
        for child in node.children:
            size += _compute_binary_node_size(child, is_64bit)
        size += (25 if is_64bit else 13)
    return size


def _write_binary_node(stream, node: FBXNode, current_offset: int, is_64bit: bool) -> int:
    node_size = _compute_binary_node_size(node, is_64bit)
    end_offset = current_offset + node_size
    num_props = len(node.properties)
    prop_bytes = node.raw_properties_data
    prop_len = len(prop_bytes)
    name_len = len(node.name)

    if is_64bit:
        stream.write(struct.pack('<QQQB', end_offset, num_props, prop_len, name_len))
    else:
        stream.write(struct.pack('<IIIB', end_offset, num_props, prop_len, name_len))

    stream.write(node.name)
    stream.write(prop_bytes)

    has_subscope = len(node.children) > 0 or node.has_sentinel
    if has_subscope:
        header_len = 25 if is_64bit else 13
        child_offset = current_offset + header_len + name_len + prop_len
        for child in node.children:
            child_offset = _write_binary_node(stream, child, child_offset, is_64bit)
        sentinel_len = 25 if is_64bit else 13
        stream.write(b'\x00' * sentinel_len)

    return end_offset


def write_binary_fbx_stream(stream, version: int, root_nodes: List[FBXNode], footer: bytes = b""):
    stream.write(BINARY_MAGIC)
    stream.write(struct.pack('<I', version))
    is_64bit = version >= 7500

    offset = stream.tell()
    for node in root_nodes:
        offset = _write_binary_node(stream, node, offset, is_64bit)

    sentinel_len = 25 if is_64bit else 13
    stream.write(b'\x00' * sentinel_len)
    if footer:
        stream.write(footer)


def write_binary_fbx_bytes(version: int, root_nodes: List[FBXNode], footer: bytes = b"") -> bytes:
    stream = io.BytesIO()
    write_binary_fbx_stream(stream, version, root_nodes, footer)
    return stream.getvalue()


def parse_binary_fbx_bytes(data: bytes) -> Tuple[int, List[FBXNode], bytes]:
    return parse_binary_fbx_stream(io.BytesIO(data))


# ==============================================================================
# ASCII FBX Representation
# ==============================================================================

class AsciiNode:
    __slots__ = ('name', 'props', 'children')

    def __init__(self, name: str, props=None, children=None):
        self.name = name
        self.props = props or []  # List of tuples: (type_tag, value_str)
        self.children = children or []

    def get_child(self, name: str) -> Optional['AsciiNode']:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def get_children(self, name: str) -> List['AsciiNode']:
        return [c for c in self.children if c.name == name]


def parse_ascii_fbx_text(text: str) -> List[AsciiNode]:
    pos = 0
    length = len(text)

    def skip_whitespace_and_comments():
        nonlocal pos
        while pos < length:
            c = text[pos]
            if c.isspace():
                pos += 1
            elif c == ';':
                while pos < length and text[pos] != '\n':
                    pos += 1
            else:
                break

    def parse_node():
        nonlocal pos
        skip_whitespace_and_comments()
        if pos >= length or text[pos] == '}':
            return None

        start = pos
        while pos < length and text[pos] not in (':', '{', '}', '\n'):
            pos += 1

        if pos >= length or text[pos] != ':':
            return None

        name = text[start:pos].strip()
        pos += 1  # skip ':'

        props = []
        while pos < length:
            skip_whitespace_and_comments()
            if pos >= length:
                break
            c = text[pos]
            if c in ('{', '\n', '}'):
                break

            if c == '"':
                pos += 1
                s_start = pos
                while pos < length and text[pos] != '"':
                    if text[pos] == '\\' and pos + 1 < length:
                        pos += 2
                    else:
                        pos += 1
                s_val = text[s_start:pos]
                if pos < length:
                    pos += 1
                props.append(('S', s_val))
            elif c == '*':
                arr_start = pos
                while pos < length and text[pos] != '{':
                    pos += 1
                if pos < length and text[pos] == '{':
                    depth = 1
                    pos += 1
                    while pos < length and depth > 0:
                        if text[pos] == '{':
                            depth += 1
                        elif text[pos] == '}':
                            depth -= 1
                        pos += 1
                arr_raw = text[arr_start:pos]
                props.append(('A', arr_raw))
            else:
                val_start = pos
                while pos < length and text[pos] not in (',', '{', '}', '\n', ';'):
                    pos += 1
                val_str = text[val_start:pos].strip()
                if val_str:
                    props.append(('V', val_str))

            skip_whitespace_and_comments()
            if pos < length and text[pos] == ',':
                pos += 1
            else:
                break

        children = []
        skip_whitespace_and_comments()
        if pos < length and text[pos] == '{':
            pos += 1
            while True:
                skip_whitespace_and_comments()
                if pos >= length or text[pos] == '}':
                    break
                child = parse_node()
                if child is None:
                    break
                children.append(child)
            if pos < length and text[pos] == '}':
                pos += 1

        return AsciiNode(name, props, children)

    root_nodes = []
    while pos < length:
        skip_whitespace_and_comments()
        if pos >= length:
            break
        node = parse_node()
        if node:
            root_nodes.append(node)
        else:
            break
    return root_nodes


def write_ascii_fbx_text(nodes: List[AsciiNode], indent=0) -> str:
    lines = []
    ind = "  " * indent
    for n in nodes:
        props_str = []
        for ptype, pval in n.props:
            if ptype == 'S':
                props_str.append(f'"{pval}"')
            elif ptype == 'A':
                props_str.append(pval)
            else:
                props_str.append(str(pval))
        p_line = ", ".join(props_str)
        if n.children:
            if p_line:
                lines.append(f"{ind}{n.name}: {p_line} {{")
            else:
                lines.append(f"{ind}{n.name}:  {{")
            lines.append(write_ascii_fbx_text(n.children, indent + 1))
            lines.append(f"{ind}}}")
        else:
            if p_line:
                lines.append(f"{ind}{n.name}: {p_line}")
            else:
                lines.append(f"{ind}{n.name}:")
    return "\n".join(lines)


# ==============================================================================
# Inspection & Cleaning Logic
# ==============================================================================

def is_fbx_binary(filepath: str) -> bool:
    with open(filepath, 'rb') as f:
        head = f.read(len(BINARY_MAGIC))
        return head == BINARY_MAGIC


def inspect_fbx(filepath: str, options: Optional[CleanOptions] = None) -> List[MeshInfo]:
    """Inspect an FBX file and return a list of meshes with their classifications."""
    if options is None:
        options = CleanOptions()

    meshes: List[MeshInfo] = []

    if is_fbx_binary(filepath):
        with open(filepath, 'rb') as f:
            version, root_nodes, _ = parse_binary_fbx_stream(f)
        objects = next((n for n in root_nodes if n.name == b"Objects"), None)
        if not objects:
            return meshes

        for child in objects.children:
            if child.name == b"Model":
                if len(child.properties) >= 2:
                    m_id = child.properties[0].value
                    raw_name = str(child.properties[1].value)
                    # Check subtype if available
                    subtype = str(child.properties[2].value) if len(child.properties) >= 3 else ""
                    # We consider meshes or objects (skip null locators unless they match collider)
                    clean = clean_node_name(raw_name)
                    cls_type, lod_idx, default_discard, reason = classify_mesh(clean, options)
                    meshes.append(MeshInfo(
                        id=m_id,
                        raw_name=raw_name,
                        clean_name=clean,
                        classification=cls_type,
                        lod_index=lod_idx,
                        default_discard=default_discard,
                        reason=reason
                    ))
    else:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            root_nodes = parse_ascii_fbx_text(f.read())
        objects = next((n for n in root_nodes if n.name == "Objects"), None)
        if not objects:
            return meshes

        for child in objects.children:
            if child.name == "Model":
                if len(child.props) >= 2:
                    try:
                        m_id = int(child.props[0][1])
                    except ValueError:
                        m_id = 0
                    raw_name = child.props[1][1]
                    clean = clean_node_name(raw_name)
                    cls_type, lod_idx, default_discard, reason = classify_mesh(clean, options)
                    meshes.append(MeshInfo(
                        id=m_id,
                        raw_name=raw_name,
                        clean_name=clean,
                        classification=cls_type,
                        lod_index=lod_idx,
                        default_discard=default_discard,
                        reason=reason
                    ))

    return meshes


def clean_fbx_file(
    input_path: str,
    output_path: str,
    options: Optional[CleanOptions] = None,
    manual_overrides: Optional[Dict[int, bool]] = None
) -> CleanReport:
    """
    Clean the input FBX file by discarding UCX colliders and LODs.
    Saves the cleaned FBX file to output_path.
    """
    if options is None:
        options = CleanOptions()
    if manual_overrides is None:
        manual_overrides = {}

    report = CleanReport(
        input_path=input_path,
        output_path=output_path,
        format="binary" if is_fbx_binary(input_path) else "ascii"
    )

    if report.format == "binary":
        _clean_binary_fbx(input_path, output_path, options, manual_overrides, report)
    else:
        _clean_ascii_fbx(input_path, output_path, options, manual_overrides, report)

    return report


def _clean_binary_fbx(
    input_path: str,
    output_path: str,
    options: CleanOptions,
    manual_overrides: Dict[int, bool],
    report: CleanReport
):
    with open(input_path, 'rb') as f:
        version, root_nodes, footer = parse_binary_fbx_stream(f)

    objects_node = next((n for n in root_nodes if n.name == b"Objects"), None)
    connections_node = next((n for n in root_nodes if n.name == b"Connections"), None)
    definitions_node = next((n for n in root_nodes if n.name == b"Definitions"), None)

    if not objects_node:
        # Nothing to clean
        with open(output_path, 'wb') as f:
            write_binary_fbx_stream(f, version, root_nodes, footer)
        return

    # Map object ids
    discarded_model_ids: Set[int] = set()
    kept_model_ids: Set[int] = set()

    for child in objects_node.children:
        if child.name == b"Model" and len(child.properties) >= 2:
            m_id = child.properties[0].value
            raw_name = str(child.properties[1].value)
            clean = clean_node_name(raw_name)
            cls_type, lod_idx, should_discard, reason = classify_mesh(clean, options)

            # Check manual override
            if m_id in manual_overrides:
                should_discard = manual_overrides[m_id]

            report.total_meshes += 1
            if should_discard:
                discarded_model_ids.add(m_id)
                if cls_type == "COLLIDER":
                    report.discarded_collider_names.append(clean)
                elif cls_type == "LOD":
                    report.discarded_lod_names.append(clean)
                else:
                    report.discarded_custom_names.append(clean)
            else:
                kept_model_ids.add(m_id)
                report.kept_mesh_names.append(clean)

    # Find connected geometries and node attributes
    # Connections: C: "OO", child_id, parent_id
    geom_to_models: Dict[int, Set[int]] = {}
    attr_to_models: Dict[int, Set[int]] = {}

    if connections_node:
        for c in connections_node.children:
            if c.name == b"C" and len(c.properties) >= 3:
                child_id = c.properties[1].value
                parent_id = c.properties[2].value
                if parent_id in discarded_model_ids or parent_id in kept_model_ids:
                    geom_to_models.setdefault(child_id, set()).add(parent_id)

    # Determine which non-model objects should be discarded
    discarded_secondary_ids: Set[int] = set()
    for child_id, parent_models in geom_to_models.items():
        # If all parent models are discarded, discard this geometry/attribute
        if parent_models.issubset(discarded_model_ids):
            discarded_secondary_ids.add(child_id)

    all_discarded_ids = discarded_model_ids.union(discarded_secondary_ids)

    # Filter Objects node
    new_objects = []
    removed_model_count = 0
    removed_geom_count = 0
    removed_attr_count = 0

    for child in objects_node.children:
        if len(child.properties) >= 1:
            obj_id = child.properties[0].value
            if obj_id in all_discarded_ids:
                if child.name == b"Model":
                    removed_model_count += 1
                elif child.name == b"Geometry":
                    removed_geom_count += 1
                elif child.name == b"NodeAttribute":
                    removed_attr_count += 1
                continue
        new_objects.append(child)

    objects_node.children = new_objects

    # Filter Connections node
    if connections_node:
        new_conns = []
        for c in connections_node.children:
            if c.name == b"C" and len(c.properties) >= 3:
                child_id = c.properties[1].value
                parent_id = c.properties[2].value
                if child_id in all_discarded_ids or parent_id in all_discarded_ids:
                    continue
            new_conns.append(c)
        connections_node.children = new_conns

    # Update Definitions counts if present
    if definitions_node:
        _update_binary_definitions(
            definitions_node,
            total_removed=len(all_discarded_ids),
            removed_models=removed_model_count,
            removed_geoms=removed_geom_count,
            removed_attrs=removed_attr_count
        )

    # Ensure target directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'wb') as f:
        write_binary_fbx_stream(f, version, root_nodes, footer)


def _update_binary_definitions(
    defs_node: FBXNode,
    total_removed: int,
    removed_models: int,
    removed_geoms: int,
    removed_attrs: int
):
    """Adjust counts in Definitions block."""
    count_node = defs_node.get_child("Count")
    if count_node and count_node.properties:
        curr = count_node.properties[0].value
        new_cnt = max(0, curr - total_removed)
        count_node.properties[0] = FBXProperty.from_int32(new_cnt)

    for ot in defs_node.get_children("ObjectType"):
        if ot.properties:
            ot_name = ot.properties[0].value
            sub_count = ot.get_child("Count")
            if sub_count and sub_count.properties:
                curr = sub_count.properties[0].value
                if ot_name == "Model":
                    sub_count.properties[0] = FBXProperty.from_int32(max(0, curr - removed_models))
                elif ot_name == "Geometry":
                    sub_count.properties[0] = FBXProperty.from_int32(max(0, curr - removed_geoms))
                elif ot_name == "NodeAttribute":
                    sub_count.properties[0] = FBXProperty.from_int32(max(0, curr - removed_attrs))


def _clean_ascii_fbx(
    input_path: str,
    output_path: str,
    options: CleanOptions,
    manual_overrides: Dict[int, bool],
    report: CleanReport
):
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        root_nodes = parse_ascii_fbx_text(f.read())

    objects_node = next((n for n in root_nodes if n.name == "Objects"), None)
    connections_node = next((n for n in root_nodes if n.name == "Connections"), None)
    definitions_node = next((n for n in root_nodes if n.name == "Definitions"), None)

    if not objects_node:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(write_ascii_fbx_text(root_nodes))
        return

    discarded_model_ids: Set[int] = set()
    kept_model_ids: Set[int] = set()

    for child in objects_node.children:
        if child.name == "Model" and len(child.props) >= 2:
            try:
                m_id = int(child.props[0][1])
            except ValueError:
                m_id = 0
            raw_name = child.props[1][1]
            clean = clean_node_name(raw_name)
            cls_type, lod_idx, should_discard, reason = classify_mesh(clean, options)

            if m_id in manual_overrides:
                should_discard = manual_overrides[m_id]

            report.total_meshes += 1
            if should_discard:
                discarded_model_ids.add(m_id)
                if cls_type == "COLLIDER":
                    report.discarded_collider_names.append(clean)
                elif cls_type == "LOD":
                    report.discarded_lod_names.append(clean)
                else:
                    report.discarded_custom_names.append(clean)
            else:
                kept_model_ids.add(m_id)
                report.kept_mesh_names.append(clean)

    # Connections
    geom_to_models: Dict[int, Set[int]] = {}
    if connections_node:
        for c in connections_node.children:
            if c.name == "C" and len(c.props) >= 3:
                try:
                    c_id = int(c.props[1][1])
                    p_id = int(c.props[2][1])
                    if p_id in discarded_model_ids or p_id in kept_model_ids:
                        geom_to_models.setdefault(c_id, set()).add(p_id)
                except ValueError:
                    pass

    discarded_secondary_ids: Set[int] = set()
    for child_id, parent_models in geom_to_models.items():
        if parent_models.issubset(discarded_model_ids):
            discarded_secondary_ids.add(child_id)

    all_discarded_ids = discarded_model_ids.union(discarded_secondary_ids)

    # Filter Objects
    new_objects = []
    removed_models = 0
    removed_geoms = 0
    for child in objects_node.children:
        if len(child.props) >= 1:
            try:
                obj_id = int(child.props[0][1])
                if obj_id in all_discarded_ids:
                    if child.name == "Model":
                        removed_models += 1
                    elif child.name == "Geometry":
                        removed_geoms += 1
                    continue
            except ValueError:
                pass
        new_objects.append(child)
    objects_node.children = new_objects

    # Filter Connections
    if connections_node:
        new_conns = []
        for c in connections_node.children:
            if c.name == "C" and len(c.props) >= 3:
                try:
                    c_id = int(c.props[1][1])
                    p_id = int(c.props[2][1])
                    if c_id in all_discarded_ids or p_id in all_discarded_ids:
                        continue
                except ValueError:
                    pass
            new_conns.append(c)
        connections_node.children = new_conns

    # Update Definitions
    if definitions_node:
        cnt_node = definitions_node.get_child("Count")
        if cnt_node and cnt_node.props:
            try:
                curr = int(cnt_node.props[0][1])
                cnt_node.props[0] = ('V', str(max(0, curr - len(all_discarded_ids))))
            except ValueError:
                pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(write_ascii_fbx_text(root_nodes))
