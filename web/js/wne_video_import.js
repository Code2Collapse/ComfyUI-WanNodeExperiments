// WNE_VideoImport — upload button: pick a video (.mp4/.mov/.webm/.mkv/.avi/.gif)
// or an .exr / image file, push it to ComfyUI/input, and select it on the node.
// (ComfyUI's built-in image-upload only accepts image/*; videos + EXR need this.)
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const ACCEPT = "video/*,.exr,.mov,.mp4,.webm,.mkv,.avi,.m4v,.gif,image/*";

function findVideoWidget(node) {
    return (node.widgets || []).find((w) => w.name === "video");
}

async function uploadFile(file) {
    const body = new FormData();
    body.append("image", file, file.name);   // ComfyUI /upload/image stores ANY file in input/
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) throw new Error(`upload failed: ${resp.status}`);
    const data = await resp.json();
    return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function addUploadWidget(node) {
    if (node._wne_vimport_wired) return;
    node._wne_vimport_wired = true;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = ACCEPT;
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", async () => {
        const f = input.files?.[0];
        if (!f) return;
        try {
            const name = await uploadFile(f);
            const w = findVideoWidget(node);
            if (w) {
                if (w.options && Array.isArray(w.options.values) && !w.options.values.includes(name)) {
                    w.options.values = [name, ...w.options.values.filter(v => !v.startsWith("("))];
                }
                w.value = name;
                w.callback?.(name);
                node.setDirtyCanvas?.(true, true);
            }
        } catch (e) {
            console.error("[WNE_VideoImport] upload error:", e);
            alert("Video upload failed: " + e.message);
        } finally {
            input.value = "";
        }
    });

    node.addWidget("button", "📁 upload video / EXR", "upload", () => input.click());

    const orig = node.onRemoved;
    node.onRemoved = function (...a) { input.remove(); return orig?.apply(this, a); };
}

app.registerExtension({
    name: "WNE.VideoImport.Upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WNE_VideoImport") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            setTimeout(() => addUploadWidget(this), 0);
            return r;
        };
    },
});
