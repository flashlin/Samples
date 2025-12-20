# T1 Web Components

A high-quality dark mode component library built with Vue 3 + TypeScript + Tailwind 3. Features intelligent Camel Case search highlighting and premium dark aesthetics.

## Core Features

- 🌙 **Dark Mode**: Native support for dark styles with a high-quality dark color palette.
- 🔍 **Camel Case Search**: Supports fast filtering via uppercase letter combinations (e.g., searching `VC` for `VueComponent`).
- 💡 **Yellow Highlighting**: Search results highlight matching characters in yellow (`text-yellow-400`).
- ⌨️ **Keyboard Support**: Full logic for arrow key selection and Enter key confirmation.

## Installation

```bash
# Using pnpm
pnpm install t1-web-components
```

## Quick Start

### 1. Import Tailwind CSS

Include Tailwind in your style file (ensure tailwindcss v3 is installed):

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 2. Use Components

```vue
<script setup>
import { DropDownList, AutoComplete } from 't1-web-components'
import { ref } from 'vue'

const selected = ref('')
const options = [
  { label: 'Vue.js', value: 'vue' },
  { label: 'TypeScript', value: 'ts' }
]
</script>

<template>
  <div class="dark p-8 bg-gray-900 min-h-screen text-white">
    <DropDownList 
      v-model="selected" 
      :options="options" 
      placeholder="Select an option..." 
    />
  </div>
</template>
```

---

## Component API

### DropDownList

A dropdown selector that enforces selection from the provided list.

#### Props

| Prop Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `modelValue` | `string \| number` | `''` | Binding value (v-model) |
| `options` | `Array<{label, value}>` | `[]` | List of options |
| `placeholder` | `string` | `'Please select...'` | Input placeholder text |
| `inputClass` | `string` | `''` | Custom CSS class for the input |

#### Events

- `@update:modelValue`: Triggered when the selected value changes.

---

### AutoComplete

An autocomplete component that allows free-text input.

#### Props

| Prop Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `modelValue` | `string \| number` | `''` | Binding value (v-model) |
| `options` | `Array<string \| {text, value}>` | `[]` | List of suggestions |
| `placeholder` | `string` | `''` | Input placeholder text |
| `inputClass` | `string` | `''` | Custom CSS class for the input |

#### Events

- `@update:modelValue`: Triggered when the input or selection changes.
- `@change`: Triggered when a list item is selected, returns the item object.

## License

[MIT LICENSE](LICENSE)
