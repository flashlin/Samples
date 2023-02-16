import type { HtmlTagDescriptor, Plugin } from "vite"

const addHtmlTagOnDevPlugin = (tags:HtmlTagDescriptor[] ): Plugin => {

  return {
    name: "vite:add-html-tag-on-dev-plugin",
    enforce: "pre",
    apply: "serve",
    transformIndexHtml: function ( html ) {
      return {
        html,
        tags
      }
    }
  }
}

export default addHtmlTagOnDevPlugin