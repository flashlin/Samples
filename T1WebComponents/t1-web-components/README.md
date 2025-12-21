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
import { DropDownList, AutoComplete, JsonEditor } from 't1-web-components'
import { ref } from 'vue'

const selected = ref('')
const options = [
  { label: 'Vue.js', value: 'vue' },
  { label: 'TypeScript', value: 'ts' }
]

const jsonData = ref('[{"id":1,"name":"John"}]')
const schema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const }
]
</script>

<template>
  <div class="dark p-8 bg-gray-900 min-h-screen text-white">
    <DropDownList
      v-model="selected"
      :options="options"
      placeholder="Select an option..."
    />

    <JsonEditor
      v-model="jsonData"
      :schema="schema"
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

---

### JsonEditor

A dynamic JSON data editor that supports both array and object editing modes. Automatically detects the mode based on input and provides a rich editing experience.

#### Props

| Prop Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `modelValue` | `string \| null` | `null` | Binding value (v-model) - must be a JSON string |
| `schema` | `JsonSchemaField[]` | `[]` | Field definitions (must use `as const` for type) |
| `compact` | `boolean` | `false` | Output compact format (single line) when true |

#### Schema Definition

Each field in the schema must follow this structure:

```typescript
interface JsonSchemaField {
  key: string                          // Field key
  label?: string                       // Display label (optional)
  type: 'string' | 'number' | 'date'  // Field type
}
```

**Important**: Use `as const` assertion for proper type inference:

```typescript
const schema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const },
  { key: 'createdAt', label: 'Created At', type: 'date' as const }
]
```

#### Events

- `@update:modelValue`: Triggered when the JSON string changes.
- `@change`: Triggered when data is modified.
- `@error`: Triggered when JSON parsing fails, returns error message.

#### Modes

JsonEditor automatically switches between two modes based on the input:

**Array Mode** (input: `[...]`)
- Displays data as a table
- Features: Search, Add, Edit, Delete, Insert, Delete All
- Changes are saved immediately

**Object Mode** (input: `{...}`)
- Displays data as a form
- Features: Edit fields, Save/Cancel buttons
- Changes require clicking Save to update

**Empty Input** (input: `""` or `null`)
- Initializes a form based on schema
- Defaults to Object Mode

#### Usage Example

**Array Mode Example:**

```vue
<script setup>
import { JsonEditor } from 't1-web-components'
import { ref } from 'vue'

const arrayData = ref('[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]')
const schema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const }
]
</script>

<template>
  <JsonEditor
    v-model="arrayData"
    :schema="schema"
    :compact="false"
  />
</template>
```

**Object Mode Example:**

```vue
<script setup>
import { JsonEditor } from 't1-web-components'
import { ref } from 'vue'

const objectData = ref('{"id":1,"name":"John","email":"john@example.com"}')
const schema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const },
  { key: 'email', label: 'Email', type: 'string' as const }
]
</script>

<template>
  <JsonEditor
    v-model="objectData"
    :schema="schema"
  />
</template>
```

**Initialize from Empty:**

```vue
<script setup>
import { JsonEditor } from 't1-web-components'
import { ref } from 'vue'

const data = ref('')
const schema = [
  { key: 'username', label: 'Username', type: 'string' as const },
  { key: 'age', label: 'Age', type: 'number' as const }
]
</script>

<template>
  <JsonEditor
    v-model="data"
    :schema="schema"
  />
</template>
```

---

## License

[MIT LICENSE](LICENSE)
