---
name: modify-vue3
description: "Modify and enhance existing Vue3 projects with TypeScript following best practices. Use when working with Vue3 (.vue) files for: (1) Implementing new features while checking for reusable code in the project, (2) Refactoring duplicated code into shared utilities, (3) Modifying components following Vue3 Composition API and TypeScript best practices, (4) Code review and optimization of Vue3 components. Always prioritize code reusability and adherence to Vue3 standards."
---

# Modify Vue3

## Overview

This skill enables systematic modification and enhancement of existing Vue3 projects with TypeScript. It emphasizes code reusability by checking existing project utilities and packages before implementing new features, refactoring duplicated code into shared functions, and adhering to Vue3 Composition API best practices.

## Workflow Decision Tree

When modifying or implementing features in a Vue3 project, follow this decision tree:

```
User Request
    ↓
1. Understand Requirements
    ↓
2. Analyze Existing Codebase
    ├─ Check project utilities/composables
    ├─ Check installed packages
    └─ Identify duplicated patterns
    ↓
3. Plan Implementation
    ├─ Use existing shared utilities? → Call them directly
    ├─ Can't reuse directly? → Refactor into shared utility first
    └─ No existing solution? → Create new shared utility
    ↓
4. Implement Following Best Practices
    ├─ Use Composition API
    ├─ Apply TypeScript properly
    └─ Follow Vue3 patterns (see references/vue3-best-practices.md)
    ↓
5. Review and Optimize
    └─ Verify code reusability and adherence to standards
```

## Step 1: Analyze Existing Codebase

Before implementing any feature, **always** analyze the existing project structure:

### Check for Reusable Code

1. **Scan common directories:**
   ```
   src/composables/     - Shared composition functions
   src/hooks/          - Custom hooks
   src/apis/           - APIs
   src/tools/          - services & Helper & Utility functions 
   src/stores/         - Pinia stores
   ```

2. **Search for similar functionality:**
   - Use grep/search to find existing implementations
   - Check if similar features already exist in other components
   - Review package.json for installed libraries that might provide the functionality

3. **Example search commands:**
   ```bash
   # Search for similar function names
   grep -r "functionName" src/
   
   # Search for similar patterns
   grep -r "pattern" src/ --include="*.vue" --include="*.ts"
   ```

### Identify Duplicated Code

Look for patterns that appear multiple times:
- Similar data fetching logic
- Repeated form validation
- Common UI interactions
- Duplicated utility functions

## Step 2: Implement with Reusability

### Strategy A: Use Existing Utilities

If a suitable utility/composable exists:

```typescript
// ✅ Good - Use existing shared utility
import { useDataFetcher } from '@/composables/useDataFetcher'

const { data, loading, error } = useDataFetcher('/api/users')
```

### Strategy B: Refactor into Shared Utility

If similar code exists but not as a shared utility:

**Before (Duplicated in multiple components):**
```typescript
// ComponentA.vue
const fetchUsers = async () => {
  loading.value = true
  try {
    const response = await fetch('/api/users')
    users.value = await response.json()
  } catch (e) {
    error.value = e
  } finally {
    loading.value = false
  }
}

// ComponentB.vue - Same pattern!
const fetchProducts = async () => {
  loading.value = true
  try {
    const response = await fetch('/api/products')
    products.value = await response.json()
  } catch (e) {
    error.value = e
  } finally {
    loading.value = false
  }
}
```

**After (Refactored into shared composable):**
```typescript
// src/composables/useApi.ts
import { ref } from 'vue'

export function useApi<T>(url: string) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)

  const execute = async () => {
    loading.value = true
    error.value = null
    try {
      const response = await fetch(url)
      data.value = await response.json()
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  return { data, loading, error, execute }
}

// ComponentA.vue & ComponentB.vue
import { useApi } from '@/composables/useApi'
const { data: users, loading, error, execute } = useApi<User[]>('/api/users')
```

### Strategy C: Create New Shared Utility

If no similar code exists, create a reusable utility from the start:

1. Place in appropriate directory (`composables/`, `utils/`, etc.)
2. Make it generic and configurable
3. Add TypeScript types
4. Document usage with JSDoc

```typescript
/**
 * Composable for managing form state with validation
 * @param initialValues - Initial form values
 * @param validationRules - Validation rules for each field
 * @returns Form state and methods
 */
export function useForm<T extends Record<string, any>>(
  initialValues: T,
  validationRules?: ValidationRules<T>
) {
  // Implementation
}
```

## Step 3: Follow Vue3 Best Practices

When implementing features, always adhere to Vue3 standards:

### Use Composition API (Not Options API)

```typescript
// ✅ Good - Composition API
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

onMounted(() => {
  console.log('Component mounted')
})
</script>

// ❌ Avoid - Options API
<script lang="ts">
export default {
  data() {
    return { count: 0 }
  },
  computed: {
    doubled() { return this.count * 2 }
  }
}
</script>
```

### Proper TypeScript Usage

```typescript
// ✅ Good - Properly typed
interface User {
  id: number
  name: string
  email: string
}

const user = ref<User | null>(null)
const users = ref<User[]>([])

// ✅ Good - Type-safe props
interface Props {
  userId: number
  mode?: 'edit' | 'view'
}

const props = defineProps<Props>()

// ✅ Good - Type-safe emits
const emit = defineEmits<{
  update: [value: string]
  delete: [id: number]
}>()
```

### Component Structure

Follow this order for better readability:

```vue
<script setup lang="ts">
// 1. Imports
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'

// 2. Props & Emits
interface Props {
  title: string
}
const props = defineProps<Props>()
const emit = defineEmits<{ submit: [data: FormData] }>()

// 3. Composables
const router = useRouter()
const { data, loading } = useApi('/api/data')

// 4. Reactive State
const isOpen = ref(false)
const formData = ref({})

// 5. Computed Properties
const isValid = computed(() => formData.value.name?.length > 0)

// 6. Methods
const handleSubmit = () => {
  emit('submit', formData.value)
}

// 7. Watchers
watch(() => props.title, (newTitle) => {
  console.log('Title changed:', newTitle)
})

// 8. Lifecycle Hooks
onMounted(() => {
  // Initialize
})
</script>

<template>
  <!-- Template -->
</template>

<style scoped>
/* Scoped styles */
</style>
```

### For detailed best practices, see:
- **references/vue3-best-practices.md** - Comprehensive Vue3 patterns and guidelines
- **references/typescript-patterns.md** - TypeScript patterns for Vue3
- **references/composables-guide.md** - How to create effective composables

## Step 4: Code Review Checklist

Before completing any modification, verify:

- [ ] Checked for existing reusable code in the project
- [ ] Refactored duplicated code into shared utilities
- [ ] Used Composition API with `<script setup>`
- [ ] Properly typed with TypeScript (props, emits, refs, etc.)
- [ ] Followed component structure guidelines
- [ ] Created reusable composables where appropriate
- [ ] Added JSDoc comments for shared utilities
- [ ] Tested the implementation

## Common Patterns

### Pattern 1: Data Fetching with Composable

Instead of repeating fetch logic, create a composable:

```typescript
// src/composables/useResourceFetcher.ts
export function useResourceFetcher<T>(endpoint: string) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)

  const fetch = async () => {
    loading.value = true
    try {
      const response = await api.get<T>(endpoint)
      data.value = response.data
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  onMounted(() => fetch())

  return { data, loading, error, refetch: fetch }
}
```

### Pattern 2: Form Management with Composable

```typescript
// src/composables/useFormValidation.ts
export function useFormValidation<T extends Record<string, any>>(
  initialData: T,
  rules: ValidationRules<T>
) {
  const formData = ref<T>(initialData)
  const errors = ref<Partial<Record<keyof T, string>>>({})
  const isValid = computed(() => Object.keys(errors.value).length === 0)

  const validate = () => {
    // Validation logic
  }

  const reset = () => {
    formData.value = { ...initialData }
    errors.value = {}
  }

  return { formData, errors, isValid, validate, reset }
}
```

### Pattern 3: Async State Management

```typescript
// src/composables/useAsyncState.ts
export function useAsyncState<T>(
  asyncFn: () => Promise<T>,
  initialValue: T
) {
  const state = ref<T>(initialValue)
  const loading = ref(false)
  const error = ref<Error | null>(null)

  const execute = async () => {
    loading.value = true
    error.value = null
    try {
      state.value = await asyncFn()
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  return { state, loading, error, execute }
}
```

## Resources

This skill includes reference documentation to support Vue3 development:

### references/

- **vue3-best-practices.md** - Comprehensive Vue3 Composition API patterns, performance optimization, and coding standards
- **typescript-patterns.md** - TypeScript patterns specifically for Vue3 projects (props typing, emits, refs, etc.)
- **composables-guide.md** - How to create effective and reusable composables

Load these references when you need detailed guidance on specific Vue3 patterns or TypeScript usage.
