Question: Whether it is new Vue({...}) in 2.x or createApp({...}) in 3.0
Answer: we need to create a Vue object instance first and assign it to a variable, 
such as vm in the example.

```
// Vue 2.x 
const vm = new Vue({ 
  data: { 
    message: 'Hello Vue 2.0!' 
  }
}); 
// Mount to the specified web node via `$mount`
vm.$mount('#app');
```

```
// Vue 3.0 
const vm = Vue.createApp({ 
  data() { 
    return { 
      message: "Hello Vue 3.0!" 
    }
  } 
}); 
vm.mount('#app');
```

Question: How to debug vitest test files in VSCode?
Answer: Open vscode, click left-bottom settings -> input "launch" -> edit in settings.json.
add a dedicated launch configuration to debug a test file in VSCode.
```
{
  "launch": {
    "configurations": [
      {
        "type": "node",
        "request": "launch",
        "name": "Debug Current Test File",
        "autoAttachChildProcesses": true,
        "skipFiles": ["<node_internals>/**", "**/node_modules/**"],
        "program": "${workspaceRoot}/node_modules/vitest/vitest.mjs",
        "args": ["run", "${relativeFile}"],
        "smartStep": true,
        "console": "integratedTerminal"
      }
    ],
    "compounds": []
  }
}
```
Then in the debug tab (Left Side Menu), ensure 'Debug Current Test File' is selected. 
You can then open the test file you want to debug and press F5 to start debugging.

Question: How to include XML files in vite?
Answer: By default, vite only includes static assets that are imported in JavaScript files. 
If you want to include XML files, you can use the assetsInclude option.

```
//vite.config.json
import { defineConfig } from 'vite';
export default defineConfig({
  // other settings ...
  assetsInclude: ['**/*.xml']
});
```

Question: How to import XML files in typescript?
Answer: By default, when using the import statement, 
TypeScript might not recognize XML file types. 
As a result, VSCode could display an error message such as Cannot find module '*.xml' or its corresponding type declarations. 
To address this, you can use the following code snippet to declare the XML file type:
```
//xml.d.ts
declare module '*.xml' {
    const content: string;
    export default content;
}
```
Create a new TypeScript declaration file (e.g., xml.d.ts) in your project's src directory. 
Then, add the provided code snippet to this file. 
This declaration informs TypeScript that XML files can be imported as strings.
After creating the declaration file, 
TypeScript should correctly recognize and allow you to import XML files using the import statement, 
and you won't encounter the "Cannot find module" error.


Question: How to create store in vue3 with Pinia?
Answer:
```
// stores/user.ts
import { defineStore } from 'pinia'

interface IUserState {
  name: string
}

const userStore = defineStore('counter', {
  state: (): IUserState => ({
    name: "",
  }),
  getters: {
    // You can define computed properties here
  },
  actions: {
    sayHello() {
      // Implementation of the action
    },
  },
})

export const useUserStore = () => userStore()
```

```
// any.vue
import { storeToRefs } from 'pinia'
import { useUserStore } from './stores/user';
const userStore = useUserStore();
// Convert the store properties to reactive references
const { name } = storeToRefs(userStore);
// Access the "sayHello" action from the store
const { sayHello } = userStore;
```
Here, you're importing the user store you defined earlier. 
You can use storeToRefs to convert the store properties to reactive references. 
This allows you to access and bind these properties in your Vue components.
