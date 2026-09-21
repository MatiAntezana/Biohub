"""Capture node creation by the frozen gap routines without editing their output."""
from contextlib import contextmanager
from functools import wraps

SYNTHETIC_STAGES = ("close_single_frame_gaps", "recover_strict_gap2")


@contextmanager
def capture_node_creation(namespace, observed_node_ids):
    """Track exact newly returned IDs for each stage, including in-place changes.

    Calls made by filter_output_graph resolve these names in its own namespace.
    No attributes/coordinates/edges are added or changed by this instrumentation.
    Restore both functions on success and failure; every dataset gets a new map.
    """
    originals = {name: namespace[name] for name in SYNTHETIC_STAGES}
    if any(not callable(fn) for fn in originals.values()):
        raise TypeError("Frozen node-creation stages must be callable")
    observed = frozenset(observed_node_ids)
    origins = {}

    def instrument(name, function):
        @wraps(function)
        def tracked(nodes_by_id, *args, **kwargs):
            before = frozenset(nodes_by_id)  # Snapshot BEFORE in-place insertion.
            result = function(nodes_by_id, *args, **kwargs)
            output_nodes, _ = result
            created = set(output_nodes) - before
            if created & (observed | set(origins)):
                raise ValueError("Node ID reused across observed/generated identities")
            origins.update({int(node_id): name for node_id in created})
            return result
        return tracked

    try:
        for name, function in originals.items():
            namespace[name] = instrument(name, function)
        yield origins
    finally:
        namespace.update(originals)
