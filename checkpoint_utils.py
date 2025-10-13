import json
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import msgpack
import numpy as np


def convert_from_msgpack(obj: Any) -> Any:
    """
    Recursively convert objects loaded from msgpack into JAX/NumPy friendly structures.
    """
    if isinstance(obj, dict):
        if "__jax_array__" in obj:
            if obj["data"] is None:
                shape = obj.get("shape")
                dtype_str = obj.get("dtype", "float32")
                if shape is not None:
                    return jnp.zeros(shape, dtype=dtype_str)
                return jnp.array(0, dtype=dtype_str)
            try:
                array_data = np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
                return jnp.array(array_data)
            except Exception:
                shape = obj.get("shape")
                dtype_str = obj.get("dtype", "float32")
                if shape is not None:
                    return jnp.zeros(shape, dtype=dtype_str)
                return jnp.array(0, dtype=dtype_str)
        if "__numpy_array__" in obj:
            try:
                return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
            except Exception:
                shape = obj.get("shape")
                dtype_str = obj.get("dtype", "float32")
                if shape is not None:
                    return np.zeros(shape, dtype=dtype_str)
                return np.array(0, dtype=dtype_str)
        return {k: convert_from_msgpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_from_msgpack(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_from_msgpack(item) for item in obj)
    return obj


def load_msgpack_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a msgpack checkpoint file and convert contained arrays into JAX/NumPy objects.
    """
    with open(checkpoint_path, "rb") as f:
        payload = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    return convert_from_msgpack(payload)


def _path_to_str(path: Tuple[Tuple[str, Any], ...]) -> str:
    if not path:
        return ""
    return "|".join(f"{kind}:{value}" for kind, value in path)


def _parse_path(path_str: str) -> List[Tuple[str, Any]]:
    if not path_str:
        return []
    out: List[Tuple[str, Any]] = []
    for part in path_str.split("|"):
        kind, value = part.split(":", 1)
        if kind in ("l", "t"):
            out.append((kind, int(value)))
        else:
            out.append((kind, value))
    return out


def _dtype_to_string(dtype: Any) -> str:
    jax_dtype = jnp.dtype(dtype)
    if jax_dtype == jnp.bfloat16:
        return "bfloat16"
    return str(np.dtype(jax_dtype))


def _string_to_dtype(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    return np.dtype(name)


def _serialize_leaf(leaf: Any) -> Tuple[np.ndarray, str, str]:
    if isinstance(leaf, jnp.ndarray):
        arr = leaf
    elif isinstance(leaf, np.ndarray):
        arr = jnp.asarray(leaf)
    else:
        arr = jnp.asarray(leaf)

    logical_dtype = _dtype_to_string(arr.dtype)
    encoding = "raw"

    if logical_dtype == "bfloat16":
        bits = jax.lax.bitcast_convert_type(arr, jnp.uint16)
        storage = np.asarray(bits, dtype=np.uint16)
        encoding = "bfloat16_bitcast"
    else:
        storage = np.asarray(arr)

    return storage, logical_dtype, encoding


def _deserialize_leaf(storage: np.ndarray, logical_dtype: str, encoding: str):
    if encoding == "bfloat16_bitcast":
        bits = jnp.asarray(storage, dtype=jnp.uint16)
        return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    target_dtype = _string_to_dtype(logical_dtype)
    return jnp.asarray(storage, dtype=target_dtype)


def _flatten_pure_structure(obj: Any):
    leaves: List[Tuple[Tuple[Tuple[str, Any], ...], Any]] = []
    containers: Dict[str, Dict[str, Any]] = {}

    def visit(node: Any, path: Tuple[Tuple[str, Any], ...]):
        if isinstance(node, dict):
            containers[_path_to_str(path)] = {"type": "dict"}
            for key in sorted(node.keys()):
                visit(node[key], path + (("d", key),))
        elif isinstance(node, list):
            containers[_path_to_str(path)] = {"type": "list", "length": len(node)}
            for idx, item in enumerate(node):
                visit(item, path + (("l", idx),))
        elif isinstance(node, tuple):
            containers[_path_to_str(path)] = {"type": "tuple", "length": len(node)}
            for idx, item in enumerate(node):
                visit(item, path + (("t", idx),))
        else:
            leaves.append((path, node))

    visit(obj, ())
    return leaves, containers


def serialize_state_to_npz_dict(pure_state: Any, step: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    leaves, containers = _flatten_pure_structure(pure_state)
    arrays: Dict[str, np.ndarray] = {}
    entries: List[Dict[str, Any]] = []

    for idx, (path, leaf) in enumerate(leaves):
        storage, dtype_str, encoding = _serialize_leaf(leaf)
        arrays[f"arr_{idx}"] = storage
        entries.append(
            {
                "path": _path_to_str(path),
                "dtype": dtype_str,
                "encoding": encoding,
            }
        )

    metadata = {
        "format": "npz_pure_v1",
        "step": int(step),
        "entries": entries,
        "containers": [
            {"path": path_str, **info} for path_str, info in containers.items()
        ],
    }
    return arrays, metadata


def save_model_state_to_npz(pure_state: Any, step: int, destination: str) -> None:
    arrays, metadata = serialize_state_to_npz_dict(pure_state, step)
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez(destination, **arrays)


def _create_container(info: Dict[str, Any]):
    container_type = info.get("type", "dict")
    if container_type == "dict":
        return {}
    length = int(info.get("length", 0))
    return [None] * length


def _assign_leaf(root: Any, path_tokens: List[Tuple[str, Any]], value: Any, container_map: Dict[str, Dict[str, Any]]):
    current = root
    for depth, (kind, key) in enumerate(path_tokens):
        is_last = depth == len(path_tokens) - 1
        prefix = path_tokens[: depth + 1]
        prefix_str = _path_to_str(prefix)

        if kind in ("d", "a"):
            if is_last:
                current[key] = value
            else:
                if key not in current or current[key] is None:
                    child_info = container_map.get(prefix_str, {"type": "dict"})
                    current[key] = _create_container(child_info)
                current = current[key]
        elif kind in ("l", "t"):
            index = int(key)
            if not isinstance(current, list):
                raise TypeError(f"Expected list-like container while assigning path {_path_to_str(path_tokens)}")
            while len(current) <= index:
                current.append(None)
            if is_last:
                current[index] = value
            else:
                if current[index] is None:
                    default_type = "tuple" if kind == "t" else "list"
                    child_info = container_map.get(prefix_str, {"type": default_type})
                    current[index] = _create_container(child_info)
                current = current[index]
        else:
            raise ValueError(f"Unsupported path kind: {kind}")


def _finalize_structure(node: Any, path: Tuple[Tuple[str, Any], ...], container_map: Dict[str, Dict[str, Any]]):
    path_str = _path_to_str(path)
    info = container_map.get(path_str)

    if info is None:
        return node

    node_type = info.get("type")

    if node_type == "dict":
        for key in list(node.keys()):
            child_path = path + (("d", key),)
            node[key] = _finalize_structure(node[key], child_path, container_map)
        return node

    if node_type in ("list", "tuple"):
        index_kind = "l" if node_type == "list" else "t"
        for idx in range(len(node)):
            child_path = path + ((index_kind, idx),)
            node[idx] = _finalize_structure(node[idx], child_path, container_map)
        if node_type == "tuple":
            return tuple(node)
        return node

    return node


def load_model_state_from_npz(npz_path: str) -> Tuple[Any, int]:
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        expected_format = metadata.get("format")
        if expected_format != "npz_pure_v1":
            raise ValueError(f"Unsupported checkpoint format: {expected_format}")

        step = metadata.get("step", 0)
        entries = metadata.get("entries", [])
        container_map = {entry["path"]: entry for entry in metadata.get("containers", [])}

        root_info = container_map.get("", {"type": "dict"})
        root = _create_container(root_info)

        for idx, entry in enumerate(entries):
            path_tokens = _parse_path(entry["path"])
            array_key = f"arr_{idx}"
            if array_key not in data:
                raise ValueError(f"Missing array {array_key} in NPZ checkpoint")
            storage = np.array(data[array_key])
            value = _deserialize_leaf(storage, entry["dtype"], entry["encoding"])
            _assign_leaf(root, path_tokens, value, container_map)

    restored = _finalize_structure(root, (), container_map)
    return restored, step
