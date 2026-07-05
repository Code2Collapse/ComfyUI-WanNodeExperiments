/**
 * ComfyUI-WanNodeExperiments — frontend extension (Phase 1).
 *
 * Tints every node belonging to this pack red (body #322 / title #533) so the
 * pack is easy to spot on the canvas. Matches by category ("WanNodeExperiments"
 * or "WanDirector") or class-name prefix ("WNE_"), so it covers the custom R&D
 * nodes, the forked Wan loaders/samplers/VAE, and the Director alike.
 *
 * Deliberately minimal: only sets colours via the standard registerExtension
 * hooks; never overlays or mutates core ComfyUI UI.
 *
 * Credits: Kijai (ComfyUI-WanVideoWrapper), wuwukaka (ComfyUI-WanAnimatePlus),
 * WhatDreamsCost (LTX Director, used with permission).
 */
import { app } from "/scripts/app.js";

const BODY_COLOR = "#322";  // red node body (simple nodes)
const TITLE_COLOR = "#533"; // red title bar (identity)
// Dark body for DOM-timeline nodes so the timeline panel blends edge-to-edge
// instead of leaving a red seam around it (matches the timeline's dark theme).
const DARK_BODY = "#181825";

function isOurNode(name, category) {
    if (name && (name.startsWith("WNE_") || name.startsWith("WanDirector"))) return true;
    if (category && (category.startsWith("WanNodeExperiments") || category.startsWith("WanDirector") ||
                     category.startsWith("C2C/Wan"))) return true;
    return false;
}

// Nodes that host a large DOM widget (timeline) — give them a dark cohesive body.
function isDomTimelineNode(name, category) {
    if (name && name.startsWith("WanDirector")) return true;
    if (category && category.startsWith("C2C/Wan")) return true;
    return false;
}

function paint(node) {
    if (!node) return;
    node.color = TITLE_COLOR; // red title kept for identification on all pack nodes
    node.bgcolor = isDomTimelineNode(node.comfyClass, node.type) ? DARK_BODY : BODY_COLOR;
}

app.registerExtension({
    name: "WanNodeExperiments.appearance",

    async beforeRegisterNodeDef(nodeType, nodeData /*, app */) {
        const name = nodeData && nodeData.name;
        const category = nodeData && nodeData.category;
        if (!isOurNode(name, category)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            paint(this);
            return r;
        };
    },

    // Recolour nodes restored from a saved workflow too.
    async loadedGraphNode(node /*, app */) {
        if (node && isOurNode(node.comfyClass, node.category)) paint(node);
    },
});

console.log(
    "%c[WanNodeExperiments]%c loaded — credits: Kijai, wuwukaka, WhatDreamsCost (LTX Director, with permission).",
    "color:#d88;font-weight:bold", "color:inherit"
);
