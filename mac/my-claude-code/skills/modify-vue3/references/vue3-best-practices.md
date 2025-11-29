# Vue3 Best Practices

## Composition API Fundamentals

### Always Use `<script setup>`

The `<script setup>` syntax provides better TypeScript inference and less boilerplate:

```vue
<!-- ✅ Preferred -->
<script setup lang="ts">
import { ref, computed } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)
</script>

<!-- ❌ Avoid -->
<script lang="ts">
import { defineComponent, ref, computed } from 'vue'

export default defineComponent({
  setup() {
    const count = ref(0)
    const doubled = computed(() => count.value * 2)
    return { count, doubled }
  }
})
</script>
```

## Reactivity Best Practices

### Use `ref()` for Primitives, `reactive()` for Objects

```typescript
// ✅ Good
const count = ref(0)
const name = ref('John')
const user = reactive({ id: 1, name: 'John' })

// ❌ Avoid - don't use reactive for primitives
const count = reactive(0) // Will not work as expected
```

### Destructuring Reactive Objects

```typescript
import { toRefs } from 'vue'

// ❌ Wrong - loses reactivity
const { name, age } = reactive({ name: 'John', age: 30 })

// ✅ Correct - maintains reactivity
const state = reactive({ name: 'John', age: 30 })
const { name, age } = toRefs(state)

// ✅ Also correct - for computed
const doubled = computed(() => count.value * 2)
const { value: doubledValue } = doubled // OK for computed
```

### Avoid Unnecessary `.value` in Templates

```vue
<script setup lang="ts">
const count = ref(0)
</script>

<template>
  <!-- ✅ Correct - automatic unwrapping -->
  <div>{{ count }}</div>
  
  <!-- ❌ Wrong - unnecessary .value -->
  <div>{{ count.value }}</div>
</template>
```

## Component Communication

### Props Definition

```typescript
// ✅ Preferred - Type-based declaration
interface Props {
  title: string
  count?: number
  items: string[]
  user: User
  mode?: 'edit' | 'view'
}

const props = withDefaults(defineProps<Props>(), {
  count: 0,
  mode: 'view'
})

// ✅ Also good - Runtime declaration with full validation
const props = defineProps({
  title: {
    type: String,
    required: true
  },
  count: {
    type: Number,
    default: 0
  }
})
```

### Emits Definition

```typescript
// ✅ Type-safe emits
const emit = defineEmits<{
  update: [value: string]
  delete: [id: number]
  submit: [data: FormData, isValid: boolean]
}>()

// Usage
emit('update', 'new value')
emit('delete', 123)
emit('submit', formData, true)

// ✅ Runtime declaration
const emit = defineEmits(['update', 'delete', 'submit'])
```

### Provide/Inject Pattern

```typescript
// Parent component
import { provide } from 'vue'

const theme = ref('dark')
provide('theme', theme)

// Child component (any level deep)
import { inject } from 'vue'

const theme = inject<Ref<string>>('theme')

// ✅ With default value and type safety
const theme = inject<Ref<string>>('theme', ref('light'))
```

## Computed Properties

### Keep Computeds Simple and Pure

```typescript
// ✅ Good - pure computation
const fullName = computed(() => `${firstName.value} ${lastName.value}`)

// ✅ Good - derived state
const hasErrors = computed(() => errors.value.length > 0)

// ❌ Avoid - side effects in computed
const fullName = computed(() => {
  console.log('Computing...') // Side effect!
  api.logAccess() // Side effect!
  return `${firstName.value} ${lastName.value}`
})

// ❌ Avoid - expensive operations without caching consideration
const sortedList = computed(() => {
  // If this runs on every render, consider useMemo pattern
  return heavyComputationFunc(largeArray.value)
})
```

### Writable Computed

```typescript
const firstName = ref('John')
const lastName = ref('Doe')

const fullName = computed({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (value: string) => {
    const parts = value.split(' ')
    firstName.value = parts[0]
    lastName.value = parts[1]
  }
})

// Usage
fullName.value = 'Jane Smith' // Sets firstName and lastName
```

## Watchers

### Use the Right Watcher Type

```typescript
// ✅ watch() - for explicit sources
watch(count, (newVal, oldVal) => {
  console.log(`Count changed from ${oldVal} to ${newVal}`)
})

// ✅ watchEffect() - for automatic dependency tracking
watchEffect(() => {
  // Automatically tracks count and doubled
  console.log(`Count: ${count.value}, Doubled: ${doubled.value}`)
})

// ✅ watch() with multiple sources
watch([firstName, lastName], ([newFirst, newLast]) => {
  console.log(`Name: ${newFirst} ${newLast}`)
})

// ✅ Immediate execution
watch(userId, async (newId) => {
  user.value = await fetchUser(newId)
}, { immediate: true })
```

### Deep Watching

```typescript
const user = reactive({ name: 'John', address: { city: 'NYC' } })

// ✅ Deep watch reactive object
watch(user, (newUser) => {
  console.log('User changed:', newUser)
}, { deep: true })

// ✅ Watch specific nested property
watch(() => user.address.city, (newCity) => {
  console.log('City changed:', newCity)
})
```

### Cleanup and Stop Watchers

```typescript
// Auto cleanup on component unmount
watch(count, (newVal) => {
  console.log(newVal)
})

// Manual stop
const stop = watch(count, (newVal) => {
  console.log(newVal)
})

// Later...
stop() // Stop watching

// Cleanup side effects
watchEffect((onCleanup) => {
  const timer = setTimeout(() => {
    console.log('Delayed log')
  }, 1000)
  
  onCleanup(() => {
    clearTimeout(timer)
  })
})
```

## Lifecycle Hooks

### Composition API Lifecycle

```typescript
import { 
  onBeforeMount, 
  onMounted, 
  onBeforeUpdate, 
  onUpdated,
  onBeforeUnmount, 
  onUnmounted 
} from 'vue'

// ✅ Correct usage
onMounted(() => {
  console.log('Component mounted')
  // Initialize, fetch data, setup listeners
})

onBeforeUnmount(() => {
  console.log('Component about to unmount')
  // Cleanup: remove listeners, cancel requests
})

// ❌ Avoid - lifecycle hooks in wrong places
const myFunction = () => {
  onMounted(() => { }) // Wrong! Must be at setup() scope
}
```

### Common Patterns

```typescript
// ✅ Data fetching on mount
onMounted(async () => {
  loading.value = true
  try {
    data.value = await fetchData()
  } catch (error) {
    handleError(error)
  } finally {
    loading.value = false
  }
})

// ✅ Event listeners with cleanup
onMounted(() => {
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
})

// ✅ Better - use watchEffect for automatic cleanup
watchEffect((onCleanup) => {
  window.addEventListener('resize', handleResize)
  onCleanup(() => {
    window.removeEventListener('resize', handleResize)
  })
})
```

## Template Best Practices

### Use `v-show` vs `v-if` Appropriately

```vue
<!-- ✅ Use v-if for conditional rendering (removes from DOM) -->
<div v-if="isLoggedIn">
  <ExpensiveComponent /> <!-- Won't be created if not logged in -->
</div>

<!-- ✅ Use v-show for toggling visibility (stays in DOM) -->
<div v-show="isVisible">
  <!-- Frequently toggled content -->
</div>
```

### Key in Lists

```vue
<!-- ✅ Always use key with v-for -->
<div v-for="item in items" :key="item.id">
  {{ item.name }}
</div>

<!-- ❌ Avoid index as key if list can be reordered -->
<div v-for="(item, index) in items" :key="index">
  {{ item.name }}
</div>
```

### Event Handling

```vue
<script setup lang="ts">
// ✅ Use inline handlers for simple logic
const count = ref(0)

// ✅ Extract complex logic into methods
const handleComplexAction = (event: Event) => {
  // Complex logic here
}
</script>

<template>
  <!-- ✅ Simple inline -->
  <button @click="count++">Increment</button>
  
  <!-- ✅ Method reference -->
  <button @click="handleComplexAction">Complex Action</button>
  
  <!-- ✅ Event modifiers -->
  <form @submit.prevent="handleSubmit">
    <input @keyup.enter="handleSearch" />
  </form>
  
  <!-- ✅ Multiple modifiers -->
  <button @click.stop.prevent="handleClick">Click Me</button>
</template>
```

## Performance Optimization

### Use `v-once` for Static Content

```vue
<template>
  <!-- ✅ Render once, never update -->
  <div v-once>
    <h1>{{ staticTitle }}</h1>
    <p>This content never changes</p>
  </div>
</template>
```

### Use `v-memo` for Expensive Renders

```vue
<template>
  <!-- ✅ Only re-render if dependencies change -->
  <div v-for="item in list" :key="item.id" v-memo="[item.id, item.selected]">
    <ExpensiveComponent :item="item" />
  </div>
</template>
```

### Lazy Load Components

```typescript
// ✅ Async component loading
import { defineAsyncComponent } from 'vue'

const HeavyComponent = defineAsyncComponent(() =>
  import('./components/HeavyComponent.vue')
)

// ✅ With loading and error states
const AsyncComponent = defineAsyncComponent({
  loader: () => import('./components/AsyncComponent.vue'),
  loadingComponent: LoadingSpinner,
  errorComponent: ErrorDisplay,
  delay: 200,
  timeout: 3000
})
```

### Optimize Large Lists

```typescript
// ✅ Use virtual scrolling for large lists
import { useVirtualList } from '@vueuse/core'

const { list, containerProps, wrapperProps } = useVirtualList(
  largeArray,
  { itemHeight: 50 }
)
```

## Composables Best Practices

### Structure and Naming

```typescript
// ✅ Name with 'use' prefix
// ✅ Single responsibility
// ✅ Return reactive state and methods

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)
  
  const increment = () => count.value++
  const decrement = () => count.value--
  const reset = () => count.value = initialValue
  
  return {
    count: readonly(count), // Prevent external mutation
    increment,
    decrement,
    reset
  }
}
```

### Async Composables

```typescript
export function useAsyncData<T>(fetcher: () => Promise<T>) {
  const data = ref<T | null>(null)
  const error = ref<Error | null>(null)
  const loading = ref(false)
  
  const execute = async () => {
    loading.value = true
    error.value = null
    try {
      data.value = await fetcher()
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }
  
  // Auto-execute on mount
  onMounted(execute)
  
  return { data, error, loading, refetch: execute }
}
```

### Composables with Cleanup

```typescript
export function useEventListener(
  target: Ref<EventTarget | null>,
  event: string,
  handler: EventListener
) {
  onMounted(() => {
    target.value?.addEventListener(event, handler)
  })
  
  onUnmounted(() => {
    target.value?.removeEventListener(event, handler)
  })
}

// Or with watchEffect for automatic cleanup
export function useEventListener2(
  target: Ref<EventTarget | null>,
  event: string,
  handler: EventListener
) {
  watchEffect((onCleanup) => {
    const element = target.value
    if (!element) return
    
    element.addEventListener(event, handler)
    onCleanup(() => {
      element.removeEventListener(event, handler)
    })
  })
}
```

## State Management

### When to Use Pinia

Use Pinia for:
- Global state shared across multiple components
- Complex state logic
- State that needs to persist
- State that needs dev tools integration

```typescript
// stores/user.ts
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', () => {
  const user = ref<User | null>(null)
  const isLoggedIn = computed(() => user.value !== null)
  
  const login = async (credentials: Credentials) => {
    const response = await api.login(credentials)
    user.value = response.user
  }
  
  const logout = () => {
    user.value = null
  }
  
  return { user, isLoggedIn, login, logout }
})
```

### Local State Management

For component-local state, use refs and reactive:

```typescript
// ✅ Simple local state
const isOpen = ref(false)
const formData = reactive({
  name: '',
  email: ''
})

// ✅ Extract to composable if shared
const { isOpen, toggle } = useToggle()
```

## Error Handling

### Async Error Handling

```typescript
// ✅ Pattern 1: Try-catch in methods
const fetchData = async () => {
  loading.value = true
  error.value = null
  try {
    data.value = await api.getData()
  } catch (e) {
    error.value = e as Error
    // Show error notification
    showNotification('Failed to fetch data', 'error')
  } finally {
    loading.value = false
  }
}

// ✅ Pattern 2: Error boundary composable
const { execute, error, loading } = useAsyncError(async () => {
  return await api.getData()
})
```

### Global Error Handler

```typescript
// main.ts
app.config.errorHandler = (err, instance, info) => {
  console.error('Global error:', err)
  console.error('Component:', instance)
  console.error('Info:', info)
  
  // Send to error tracking service
  errorTracker.captureException(err)
}
```

## Testing Considerations

### Make Components Testable

```typescript
// ✅ Extract business logic to composables
export function useUserForm() {
  const formData = reactive({ name: '', email: '' })
  const errors = ref<Record<string, string>>({})
  
  const validate = () => {
    errors.value = {}
    if (!formData.name) errors.value.name = 'Name is required'
    if (!formData.email) errors.value.email = 'Email is required'
    return Object.keys(errors.value).length === 0
  }
  
  return { formData, errors, validate }
}

// Component uses the composable
const { formData, errors, validate } = useUserForm()

// Now composable can be tested independently
```

### Avoid Tight Coupling

```typescript
// ❌ Tightly coupled to specific implementation
const fetchUsers = async () => {
  const response = await axios.get('/api/users')
  return response.data
}

// ✅ Dependency injection for testability
export function useUsers(api: ApiClient) {
  const fetchUsers = async () => {
    return await api.get('/users')
  }
  
  return { fetchUsers }
}
```
