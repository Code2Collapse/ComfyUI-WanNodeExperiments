// wne_lipsync_progress.js — native step-progress bar for the Audio/LipSync
// suite (Kijai-style: thin bar + step label under the title, no DOM overlay).
//
// Backend sends `wne.lipsync.progress` {node, step, pct, label} via
// PromptServer.send_sync; we stash it on the node and paint in
// onDrawForeground (graph-space → zoom/pan safe automatically). The bar
// clears itself 2.5s after completion.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TARGETS = new Set(["WNE_AdvancedAudioSeparator", "WNE_InfiniteTalkV2V"]);

if (!(app.extensions || []).some((e) => e?.name === "WNE.LipSyncProgress")) app.registerExtension({
    name: "WNE.LipSyncProgress",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGETS.has(nodeData?.name)) return;
        const onDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDraw ? onDraw.apply(this, arguments) : undefined;
            const p = this.__wneProg;
            if (!p) return r;
            if (p.pct >= 1 && performance.now() - p.t > 2500) { this.__wneProg = null; return r; }
            const w = this.size[0] - 20, x = 10, y = 4;
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.fillRect(x, y, w, 12);
            ctx.fillStyle = p.pct >= 1 ? "#8cff66" : "#4da6ff";
            ctx.fillRect(x + 1, y + 1, Math.max(2, (w - 2) * Math.min(1, p.pct)), 10);
            ctx.fillStyle = "rgba(240,244,255,0.95)";
            ctx.font = "9px ui-monospace,monospace";
            ctx.textAlign = "left"; ctx.textBaseline = "middle";
            let txt = `${p.step}: ${p.label}`;
            if (ctx.measureText(txt).width > w - 8) txt = txt.slice(0, 42) + "…";
            ctx.fillText(txt, x + 4, y + 6);
            ctx.restore();
            return r;
        };
    },
    async setup() {
        api.addEventListener("wne.lipsync.progress", (ev) => {
            const d = ev.detail || {};
            const node = app.graph?._nodes?.find((n) => String(n.id) === String(d.node));
            if (!node) return;
            node.__wneProg = { step: d.step || "", pct: +d.pct || 0,
                               label: d.label || "", t: performance.now() };
            app.graph.setDirtyCanvas(true, false);
        });
    },
});
