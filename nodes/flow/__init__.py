"""
flow — optical-flow consistency stack for Wan vid2vid editing.

Foundation (RAFT flow + warp + occlusion) that the FlowVid / FRESCO / TokenFlow
adaptations build on. See flow_core.py for the grounded math and flow_nodes.py
for the WNE nodes.
"""
from __future__ import annotations

try:
    from .flow_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001 - never hard-crash the pack on a flow import error
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}
