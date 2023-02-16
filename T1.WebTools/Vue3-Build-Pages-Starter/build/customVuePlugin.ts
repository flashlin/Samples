import { default as baseVue } from "@vitejs/plugin-vue"


export default () => baseVue({
  template: {
    compilerOptions: {
      nodeTransforms: [
        (node) => {
          if (process.env.NODE_ENV === "production") {
            if (node.type === 1 /*NodeTypes.ELEMENT*/) {
              for (let i = 0; i < node.props.length; i++) {
                const p = node.props[i]
                if (p && p.type === 6/*NodeTypes.ATTRIBUTE*/ && p.name === "data-testid") {
                  node.props.splice(i, 1)
                  i--
                }
              }
            }
          }
        },
      ],
    },
  },
})