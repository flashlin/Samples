import { default as baseVue } from "@vitejs/plugin-vue";

const NodeTypes = {
  ELEMENT: 1,
  ATTRIBUTE: 6
};

export default () => baseVue({
  template: {
    compilerOptions: {
      nodeTransforms: [
        (node: any) => {
          if (process.env.NODE_ENV === "production") {
            if (node.type === NodeTypes.ELEMENT ) {
              for (let i = 0; i < node.props.length; i++) {
                const p = node.props[i];
                if (p && p.type === NodeTypes.ATTRIBUTE && p.name === "data-test-id") {
                  node.props.splice(i, 1);
                  i--;
                }
              }
            }
          }
        },
      ],
    },
  },
});