"""R11 — registration smoke under ComfyUI's real loader.

This repo had ZERO tests while registering 40 nodes. The class of bug this file
exists to catch has now hit this workspace four times: a pack that imports fine
under pytest and registers NOTHING in production. Most recently ComfyUI-GLM_Image
shipped at 0 nodes with 6 green tests, because pytest puts the repo root on
sys.path and ComfyUI does not.

Loader mirrors `third_party/ComfyUI/nodes.py:2243-2263`:

    sys_module_name = module_path.replace(".", "_x_")          # :2250
    module_spec = importlib.util.spec_from_file_location(...)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[sys_module_name] = module                      # :2262  BEFORE exec
    module_spec.loader.exec_module(module)                     # :2263

Two details are load-bearing, both learned the hard way:
  * the sys.modules assignment BEFORE exec_module, without which the package's
    own relative imports cannot resolve;
  * STRIPPING the pack directory from sys.path for the duration of the load.
    Without that strip an earlier test in the same session leaves the root on
    sys.path, an absolute sibling import resolves, and this test passes while
    production is broken — verified on GLM_Image, where the bug was deliberately
    reintroduced and this test still passed until the strip was added.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Core first-party nodes. The pack also attempts vendored wanwrapper/animateplus
#: loads which register extra WNE_ / WNE_AP_ ids when those trees are present, so
#: this is a FLOOR, not an equality — asserting equality would fail on machines
#: that have the optional trees.
EXPECTED_MIN_NODES = 40

#: A sample that must always be present. Chosen across subsystems so a partial
#: import failure in any one of them is caught.
REQUIRED_NODE_IDS = {
    "WNE_RAFTOpticalFlow",
    "WNE_FlowTemporalConsistency",
    "WNE_WanModelLoader",
    "WNE_WanT5GemmaLoader",
    "WNE_AdvancedAudioSeparator",
    "WNE_WanVAEEncode",
    "WanDirectorC2C",
}


def _load_pack_like_comfyui():
    module_path = str(ROOT)
    sys_module_name = module_path.replace(".", "_x_")

    saved_path = list(sys.path)
    saved_mod = sys.modules.get(sys_module_name)
    # Production has ComfyUI's OWN modules importable (folder_paths, comfy.*) but
    # NOT the pack directory on sys.path. Reproduce exactly that: add the former,
    # strip the latter. Omitting ComfyUI under-reports — nodes that legitimately
    # import folder_paths fail to register and the test blames the pack.
    comfy_root = ROOT.parent / "third_party" / "ComfyUI"
    if comfy_root.is_dir() and str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
    sys.modules.pop(sys_module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(
            sys_module_name, ROOT / "__init__.py", submodule_search_locations=[module_path]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[sys_module_name] = module  # BEFORE exec_module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        if saved_mod is not None:
            sys.modules[sys_module_name] = saved_mod


def test_pack_registers_nodes_under_the_real_loader():
    pack = _load_pack_like_comfyui()
    mappings = getattr(pack, "NODE_CLASS_MAPPINGS", None)
    assert mappings is not None, "__init__.py exposes no NODE_CLASS_MAPPINGS"
    assert len(mappings) >= EXPECTED_MIN_NODES, (
        f"registered {len(mappings)} nodes, expected at least {EXPECTED_MIN_NODES}. "
        f"A drop here means an import is failing silently: {sorted(mappings)}"
    )


def test_core_node_ids_all_present():
    pack = _load_pack_like_comfyui()
    ids = set(getattr(pack, "NODE_CLASS_MAPPINGS", {}))
    missing = REQUIRED_NODE_IDS - ids
    assert not missing, f"nodes failed to register: {sorted(missing)}"


def test_every_registered_node_declares_a_contract():
    """No stub nodes: each must declare its inputs, outputs and entrypoint."""
    pack = _load_pack_like_comfyui()
    problems = []
    for node_id, cls in getattr(pack, "NODE_CLASS_MAPPINGS", {}).items():
        if not hasattr(cls, "INPUT_TYPES"):
            problems.append(f"{node_id}: no INPUT_TYPES")
        if not getattr(cls, "RETURN_TYPES", None) and not getattr(cls, "OUTPUT_NODE", False):
            problems.append(f"{node_id}: no RETURN_TYPES and not an OUTPUT_NODE")
        fn = getattr(cls, "FUNCTION", None)
        if fn and not hasattr(cls, fn):
            problems.append(f"{node_id}: FUNCTION={fn!r} but no such method")
    assert not problems, "incomplete node contracts:\n  " + "\n  ".join(problems)


def test_display_names_cover_every_node():
    pack = _load_pack_like_comfyui()
    ids = set(getattr(pack, "NODE_CLASS_MAPPINGS", {}))
    names = set(getattr(pack, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    missing = ids - names
    assert not missing, f"registered without a display name: {sorted(missing)}"


def test_no_module_named_utils_or_nodes_py():
    """Names that collide with ComfyUI's own top-level modules.

    ComfyUI imports its `utils` package and `nodes` module at startup, before
    custom nodes load. A same-named module at this pack's root becomes
    unreachable, which registered ZERO nodes in ComfyUI-MiniMaxSuite until it
    was renamed. `nodes/` here is a PACKAGE, imported relatively, so it is fine —
    this guards a top-level `nodes.py` / `utils.py` FILE appearing beside it.
    """
    # NOTE: a 34 KB vendored `utils.py` (Kijai WanVideoWrapper, Apache-2.0) DOES
    # sit at this pack root. Nothing imports it bare, so it is dormant rather than
    # broken — deleting vendored code is not this test's call. The live hazard is
    # an actual bare `import utils` / `from utils import ...`, which in production
    # resolves to ComfyUI's own package and fails. That is what is asserted.
    offenders = []
    for py in list(ROOT.glob("*.py")) + list((ROOT / "nodes").rglob("*.py")):
        for n, line in enumerate(py.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            st = line.strip()
            if st in ("import utils", "import nodes") or st.startswith(
                ("from utils import", "from nodes import")
            ):
                offenders.append(f"{py.relative_to(ROOT)}:{n}: {st}")
    assert not offenders, (
        "bare import of a name ComfyUI owns at top level; use a relative import:\n  "
        + "\n  ".join(offenders)
    )


def test_web_directory_points_at_a_real_folder():
    pack = _load_pack_like_comfyui()
    web = getattr(pack, "WEB_DIRECTORY", None)
    if web is None:
        pytest.skip("pack declares no WEB_DIRECTORY")
    resolved = (ROOT / str(web).lstrip("./")).resolve()
    assert resolved.is_dir(), f"WEB_DIRECTORY={web!r} does not exist ({resolved})"
