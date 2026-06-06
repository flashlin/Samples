import { defineConfig } from "wxt"

export default defineConfig({
  manifest: {
    name: "PageProbe",
    description: "Inspect and automate the current browser pages through a local gateway.",
    version: "0.1.0",
    minimum_chrome_version: "125",
    permissions: ["debugger", "tabs", "storage", "scripting"],
    host_permissions: ["<all_urls>"]
  }
})
