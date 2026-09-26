/**
 * ComfyUI-WanNodeExperiments - frontend entry point.
 *
 * This file used to paint every node of the pack red (title #533, body #322).
 * It did so unconditionally, and ALSO from loadedGraphNode - so a colour the
 * user set on a WNE node was overwritten every time a workflow was loaded.
 *
 * The pack now carries the Code2Collapse house look instead, from
 * ../_c2c_brand.js - the same night title bar and body as every other
 * Code2Collapse pack, applied only where a node has no colour yet, so a saved
 * or user-chosen colour always wins. The Director's timeline nodes, which
 * wanted a dark body so the panel blends edge to edge, get it from the same
 * brand body.
 *
 * Credits: Kijai (ComfyUI-WanVideoWrapper), wuwukaka (ComfyUI-WanAnimatePlus),
 * WhatDreamsCost (LTX Director, used with permission).
 */

console.log(
    "%c[WanNodeExperiments]%c loaded — credits: Kijai, wuwukaka, WhatDreamsCost (LTX Director, with permission).",
    "color:#d88;font-weight:bold", "color:inherit"
);
