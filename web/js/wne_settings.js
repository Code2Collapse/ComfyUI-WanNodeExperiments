// WanNodeExperiments settings (L3.02; design docs/research/L3_settings_secrets_audit.md S9 in the C2C workspace).
// Declared here so they exist when the pack is installed alone; the backend reads the saved value through
// _c2c_saved_setting.py. The environment variable WNE_PROXY_CACHE_MB still wins when it is set.
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "C2C.WNE.Settings",
    settings: [
        {
            id: "c2c.wne.proxyCacheMB",
            name: "Video proxy cache size (MB)",
            category: ["C2C", "WNE", "Video", "proxy cache"],
            type: "number",
            defaultValue: 2048,
            attrs: { min: 64, max: 1000000, step: 64 },
            tooltip: "Disk space the Video Import preview proxies may use; the oldest proxies are removed first. "
                + "The environment variable WNE_PROXY_CACHE_MB overrides this when it is set.",
        },
    ],
});
