// WanDirectorExtraArgs — show each advanced block's parameters ONLY when its
// toggle is on (hide-when-false). Keeps the companion node tidy: by default the
// enables are off, so the node is short; flip one on and its params appear.
// (Same rule the user wants across every node.)
import { app } from "/scripts/app.js";

// gated widget -> predicate over the current widget values
const GATES = {
    prompt_relay_epsilon:        (v) => !!v.enable_prompt_relay,
    guidance_rescale_phi:        (v) => !!v.enable_dynamic_cfg,
    phase_shift_pct:             (v) => !!v.enable_phase_shift,
    structure_prompt:            (v) => !!v.enable_multi_clip,
    detail_prompt:               (v) => !!v.enable_multi_clip,
    nag_scale:                   (v) => !!v.enable_nag,
    asymflow_shift:              (v) => !!v.enable_asymflow,
    cache_threshold:             (v) => String(v.cache_type ?? "none") !== "none",
    slg_layers:                  (v) => !!v.enable_slg,
    slg_scale:                   (v) => !!v.enable_slg,
    feta_scale:                  (v) => !!v.enable_feta,
    riflex_k:                    (v) => !!v.enable_riflex,
    // EverAnimate block: its params matter only when a real stage is chosen
    everanimate_num_chunks:      (v) => !/^(off|none|disabled)$/i.test(String(v.everanimate_stage ?? "")),
    everanimate_overlap_frames:  (v) => !/^(off|none|disabled)$/i.test(String(v.everanimate_stage ?? "")),
    everanimate_lora_strength:   (v) => !/^(off|none|disabled)$/i.test(String(v.everanimate_stage ?? "")),
    everanimate_anchor_strategy: (v) => !/^(off|none|disabled)$/i.test(String(v.everanimate_stage ?? "")),
};

const TRIGGERS = ["enable_prompt_relay", "enable_dynamic_cfg", "enable_phase_shift",
    "enable_multi_clip", "enable_nag", "enable_asymflow", "cache_type",
    "enable_slg", "enable_feta", "enable_riflex", "everanimate_stage"];

function setHidden(w, hidden) {
    if (!w) return;
    if (hidden) {
        if (!("__wne_t" in w)) { w.__wne_t = w.type; w.__wne_cs = w.computeSize; }
        w.type = "hidden"; w.computeSize = () => [0, -4]; w.hidden = true;
        if (w.element) { if (!("__wne_d" in w)) w.__wne_d = w.element.style.display; w.element.style.display = "none"; }
    } else {
        if ("__wne_t" in w) { w.type = w.__wne_t; delete w.__wne_t; }
        if ("__wne_cs" in w) { const cs = w.__wne_cs; if (cs === undefined) delete w.computeSize; else w.computeSize = cs; delete w.__wne_cs; }
        w.hidden = false;
        if (w.element) { w.element.style.display = ("__wne_d" in w) ? (w.__wne_d ?? "") : ""; delete w.__wne_d; }
    }
}

function apply(node) {
    const vals = {};
    for (const w of node.widgets || []) vals[w.name] = w.value;
    for (const w of node.widgets || []) {
        const g = GATES[w.name];
        if (g) setHidden(w, !g(vals));
    }
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hook(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => { const r = orig?.call(w, v, ...rest); apply(node); return r; };
}

app.registerExtension({
    name: "WNE.WanDirectorExtraArgs.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanDirectorExtraArgs") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            for (const n of TRIGGERS) hook(this, n);
            setTimeout(() => apply(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => apply(this), 0);
            return r;
        };
    },
});
