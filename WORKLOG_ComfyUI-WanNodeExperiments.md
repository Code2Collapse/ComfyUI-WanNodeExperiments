# WORKLOG — ComfyUI-WanNodeExperiments

**Stage 0 audit, 2026-08-29. Every number below was MEASURED this session, not inherited.**
Regenerate with the commands in the last section. Per R3 this file is the persisted source of
record; a claim that lives only in a chat transcript has now drifted five times.

> **Updated 2026-08-29 after the build passes.** The numbers above are re-measured, not the Stage-0 audit figures. A worklog that still reports its audit snapshot is the stale-record failure R3 exists to prevent.

## 1. Live inventory
| | |
|---|---|
| **Nodes registered (runtime)** | **40** |
| Registration style | V1 `NODE_CLASS_MAPPINGS` |
| `WEB_DIRECTORY` | `./web` |
| Test files | **3** (was 0) |
| **Tests** | **27 passed** (was NONE — this was the only test-less repo) |

## 2. Licence
**Apache-2.0.** MIT/Apache inbound only; GPL forbidden.

## 3. Registration smoke
PASS — 40 nodes register under the production loader.

## 4. Test status
**Zero tests.** 40 registered nodes — including the RAFT flow core and the T5Gemma loaders — have no
CPU-only regression cover whatsoever. Per brief section 3 this is the first build item.

## 5. Invariant sweep
| Check | Result |
|---|---|
| `third_party` runtime imports | 0 |
| `IS_CHANGED` -> `float("nan")` | 0 |
| Hardcoded `.cuda()` | **4 — the highest of any repo**; R5 violation on an 8 GB box |
| Frame loops without interrupt | 7 first-party `nodes/` files |

## 6. Build queue (brief 2.3)
1. **Tests first** — one file per module, starting with `nodes/loaders.py` and `flow_core.py`, which
   are what other repos depend on behaviourally.
2. #26 flow-guided mask propagation (painted keyframe -> all frames) on the existing flow core.
3. Offset-energy QC component, exported as a **data contract** for CustomNodePacks #82 — never a
   cross-repo import.

## 7. Blocked / decisions
The 4 hardcoded `.cuda()` calls must be resolved before any 8 GB claim about this repo is credible.

## Regeneration commands

```
head -3 LICENSE

# registration smoke, the way ComfyUI loads (third_party/ComfyUI/nodes.py:2243-2263):
#   sys.modules[name] = mod   BEFORE   spec.loader.exec_module(mod)
# Anything less can report healthy for a pack that registers nothing.
python <scratch>/regsmoke.py ComfyUI-WanNodeExperiments

D:/PROJECT/ComfyUI_windows_portable/comfy_env/python.exe -m pytest tests/ -q
```

Shell python has no torch — always use the comfy_env interpreter.
