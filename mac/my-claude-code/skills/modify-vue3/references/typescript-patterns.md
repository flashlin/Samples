# TypeScript Patterns for Vue3

## Component Props Typing

### Basic Props Definition

```typescript
// ✅ Type-based (Recommended)
interface Props {
  title: string
  count: number
  isActive?: boolean
  items: string[]
  user: User
  callback?: (value: string) => void
}

const props = defineProps<Props>()

// ✅ With defaults
const props = withDefaults(defineProps<Props>(), {
  count: 0,
  isActive: false,
  items: () => [],
  callback: undefined
})
```

### Complex Props Types

```typescript
// Union types
interface Props {
  status: 'pending' | 'success' | 'error'
  size: 'small' | 'medium' | 'large'
}

// Generic props
interface Props<T = any> {
  data: T
  onSelect: (item: T) => void
}

// Optional with specific types
interface Props {
  config?: {
    theme: string
    locale: string
  }
}

// Array of specific type
interface Props {
  users: Array<{
    id: number
    name: string
    role: 'admin' | 'user'
  }>
}
```

### Props Validation with Runtime

```typescript
// When you need both type checking AND runtime validation
import { PropType } from 'vue'

interface User {
  id: number
  name: string
}

const props = defineProps({
  user: {
    type: Object as PropType<User>,
    required: true,
    validator: (value: User) => {
      return value.id > 0 && value.name.length > 0
    }
  },
  status: {
    type: String as PropType<'pending' | 'success' | 'error'>,
    default: 'pending',
    validator: (value: string) => {
      return ['pending', 'success', 'error'].includes(value)
    }
  }
})
```

## Emits Typing

### Basic Emits

```typescript
// ✅ Type-safe emits definition
const emit = defineEmits<{
  // Event with single parameter
  update: [value: string]
  
  // Event with multiple parameters
  submit: [data: FormData, isValid: boolean]
  
  // Event with no parameters
  close: []
  
  // Event with optional parameter
  change: [value?: number]
}>()

// Usage
emit('update', 'new value')
emit('submit', formData, true)
emit('close')
emit('change', 42)
emit('change') // Also valid
```

### Events with Object Payloads

```typescript
interface UpdatePayload {
  field: string
  value: any
  timestamp: number
}

const emit = defineEmits<{
  update: [payload: UpdatePayload]
  delete: [id: number]
}>()

// Usage
emit('update', {
  field: 'name',
  value: 'John',
  timestamp: Date.now()
})
```

### Generic Emits

```typescript
// For reusable components
interface Props<T> {
  items: T[]
}

const emit = defineEmits<{
  select: [item: T]
  change: [items: T[]]
}>()
```

## Refs Typing

### Basic Refs

```typescript
import { ref, Ref } from 'vue'

// ✅ Explicit type
const count = ref<number>(0)
const name = ref<string>('John')
const isActive = ref<boolean>(false)

// ✅ Nullable types
const user = ref<User | null>(null)
const data = ref<Data | undefined>(undefined)

// ✅ Array types
const items = ref<string[]>([])
const users = ref<User[]>([])

// ✅ Complex types
interface FormData {
  name: string
  email: string
  age: number
}

const formData = ref<FormData>({
  name: '',
  email: '',
  age: 0
})
```

### Refs with Type Inference

```typescript
// ✅ Type is inferred
const count = ref(0) // Ref<number>
const name = ref('John') // Ref<string>
const items = ref([1, 2, 3]) // Ref<number[]>

// ⚠️ Be careful with empty arrays
const items = ref([]) // Ref<never[]> - not useful!
const items = ref<number[]>([]) // ✅ Explicit type needed
```

### Template Refs

```typescript
import { ref, onMounted } from 'vue'

// ✅ DOM element ref
const inputRef = ref<HTMLInputElement | null>(null)
const divRef = ref<HTMLDivElement | null>(null)

onMounted(() => {
  inputRef.value?.focus()
  console.log(divRef.value?.offsetHeight)
})

// ✅ Component ref
import MyComponent from './MyComponent.vue'

const componentRef = ref<InstanceType<typeof MyComponent> | null>(null)

onMounted(() => {
  // Access component methods/properties
  componentRef.value?.someMethod()
})
```

```vue
<template>
  <input ref="inputRef" />
  <div ref="divRef">Content</div>
  <MyComponent ref="componentRef" />
</template>
```

## Reactive Typing

### Basic Reactive

```typescript
import { reactive } from 'vue'

// ✅ Interface definition
interface User {
  id: number
  name: string
  email: string
  settings?: {
    theme: string
    notifications: boolean
  }
}

const user = reactive<User>({
  id: 1,
  name: 'John',
  email: 'john@example.com'
})

// ✅ Type is inferred from initial value
const state = reactive({
  count: 0,
  name: 'John'
}) // { count: number, name: string }
```

### Reactive with Arrays

```typescript
interface Item {
  id: number
  name: string
}

// ✅ Reactive array
const items = reactive<Item[]>([])

// Add items
items.push({ id: 1, name: 'Item 1' })

// ⚠️ Don't reassign reactive arrays
// items = [] // ❌ Loses reactivity!

// ✅ Instead, use splice or clear
items.splice(0, items.length)
// Or
items.length = 0
```

### Complex Reactive State

```typescript
interface AppState {
  user: User | null
  settings: {
    theme: 'light' | 'dark'
    language: string
  }
  notifications: Array<{
    id: string
    message: string
    type: 'info' | 'warning' | 'error'
  }>
  loading: boolean
}

const state = reactive<AppState>({
  user: null,
  settings: {
    theme: 'light',
    language: 'en'
  },
  notifications: [],
  loading: false
})
```

## Computed Typing

### Basic Computed

```typescript
import { ref, computed } from 'vue'

const count = ref(0)

// ✅ Type is inferred
const doubled = computed(() => count.value * 2) // ComputedRef<number>

// ✅ Explicit type (rarely needed)
const doubled = computed<number>(() => count.value * 2)
```

### Computed with Complex Types

```typescript
interface User {
  firstName: string
  lastName: string
  age: number
}

const user = ref<User>({
  firstName: 'John',
  lastName: 'Doe',
  age: 30
})

// ✅ Type is inferred from return value
const fullName = computed(() => {
  return `${user.value.firstName} ${user.value.lastName}`
}) // ComputedRef<string>

const isAdult = computed(() => {
  return user.value.age >= 18
}) // ComputedRef<boolean>

// ✅ Computed returning objects
interface DisplayUser {
  name: string
  isAdult: boolean
}

const displayUser = computed<DisplayUser>(() => ({
  name: `${user.value.firstName} ${user.value.lastName}`,
  isAdult: user.value.age >= 18
}))
```

### Writable Computed

```typescript
const firstName = ref('John')
const lastName = ref('Doe')

// ✅ Writable computed with proper typing
const fullName = computed<string>({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (value: string) => {
    const [first, last] = value.split(' ')
    firstName.value = first
    lastName.value = last
  }
})
```

## Composables Typing

### Basic Composable

```typescript
import { ref, Ref } from 'vue'

// ✅ Export return type for reusability
export interface UseCounterReturn {
  count: Ref<number>
  increment: () => void
  decrement: () => void
  reset: () => void
}

export function useCounter(initialValue = 0): UseCounterReturn {
  const count = ref(initialValue)
  
  const increment = () => count.value++
  const decrement = () => count.value--
  const reset = () => count.value = initialValue
  
  return {
    count,
    increment,
    decrement,
    reset
  }
}

// Usage
const { count, increment } = useCounter(10)
```

### Generic Composables

```typescript
// ✅ Generic composable with constraints
export interface UseAsyncReturn<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<Error | null>
  execute: () => Promise<void>
}

export function useAsync<T>(
  asyncFn: () => Promise<T>
): UseAsyncReturn<T> {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)
  
  const execute = async () => {
    loading.value = true
    error.value = null
    try {
      data.value = await asyncFn()
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }
  
  return { data, loading, error, execute }
}

// Usage with type inference
const { data, loading, error, execute } = useAsync<User[]>(
  () => fetch('/api/users').then(r => r.json())
)
// data is Ref<User[] | null>
```

### Composable with Options

```typescript
interface UseFetchOptions<T> {
  immediate?: boolean
  onSuccess?: (data: T) => void
  onError?: (error: Error) => void
  transform?: (data: any) => T
}

export function useFetch<T = any>(
  url: string,
  options: UseFetchOptions<T> = {}
) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)
  
  const execute = async () => {
    loading.value = true
    error.value = null
    try {
      const response = await fetch(url)
      let result = await response.json()
      
      if (options.transform) {
        result = options.transform(result)
      }
      
      data.value = result
      options.onSuccess?.(result)
    } catch (e) {
      const err = e as Error
      error.value = err
      options.onError?.(err)
    } finally {
      loading.value = false
    }
  }
  
  if (options.immediate !== false) {
    onMounted(execute)
  }
  
  return { data, loading, error, execute }
}

// Usage
const { data } = useFetch<User>('/api/user', {
  immediate: true,
  transform: (raw) => ({
    id: raw.id,
    name: raw.full_name
  }),
  onSuccess: (user) => {
    console.log('User loaded:', user.name)
  }
})
```

## Async/Await Typing

### Async Functions

```typescript
// ✅ Properly typed async functions
const fetchUser = async (id: number): Promise<User> => {
  const response = await fetch(`/api/users/${id}`)
  return await response.json()
}

const fetchUsers = async (): Promise<User[]> => {
  const response = await fetch('/api/users')
  return await response.json()
}

// ✅ With error handling
const fetchUserSafe = async (id: number): Promise<User | null> => {
  try {
    const response = await fetch(`/api/users/${id}`)
    return await response.json()
  } catch (error) {
    console.error('Failed to fetch user:', error)
    return null
  }
}
```

### Async Composables

```typescript
export function useAsyncData<T>(fetcher: () => Promise<T>) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)
  
  const execute = async (): Promise<void> => {
    loading.value = true
    error.value = null
    try {
      data.value = await fetcher()
    } catch (e) {
      error.value = e as Error
      throw e // Re-throw if needed
    } finally {
      loading.value = false
    }
  }
  
  return { data, loading, error, execute }
}
```

## Type Guards and Narrowing

### Basic Type Guards

```typescript
interface User {
  id: number
  name: string
  email: string
}

interface Admin extends User {
  role: 'admin'
  permissions: string[]
}

// ✅ Type guard function
function isAdmin(user: User | Admin): user is Admin {
  return 'role' in user && user.role === 'admin'
}

// Usage
const user = ref<User | Admin>(/* ... */)

if (isAdmin(user.value)) {
  // TypeScript knows user.value is Admin here
  console.log(user.value.permissions)
}
```

### Discriminated Unions

```typescript
interface LoadingState {
  status: 'loading'
}

interface SuccessState<T> {
  status: 'success'
  data: T
}

interface ErrorState {
  status: 'error'
  error: Error
}

type AsyncState<T> = LoadingState | SuccessState<T> | ErrorState

const state = ref<AsyncState<User>>({ status: 'loading' })

// ✅ TypeScript narrows based on status
if (state.value.status === 'success') {
  // state.value.data is available and typed as User
  console.log(state.value.data.name)
} else if (state.value.status === 'error') {
  // state.value.error is available
  console.log(state.value.error.message)
}
```

## Event Handlers Typing

### DOM Events

```typescript
// ✅ Properly typed event handlers
const handleClick = (event: MouseEvent) => {
  console.log('Clicked at:', event.clientX, event.clientY)
}

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  console.log('Input value:', target.value)
}

const handleKeyPress = (event: KeyboardEvent) => {
  if (event.key === 'Enter') {
    console.log('Enter pressed')
  }
}

const handleSubmit = (event: SubmitEvent) => {
  event.preventDefault()
  const form = event.target as HTMLFormElement
  const formData = new FormData(form)
  // Process form data
}
```

```vue
<template>
  <button @click="handleClick">Click Me</button>
  <input @input="handleInput" />
  <input @keypress="handleKeyPress" />
  <form @submit="handleSubmit">
    <!-- form content -->
  </form>
</template>
```

### Custom Events

```typescript
// Component A - Emitting custom events
interface UpdatePayload {
  field: string
  value: any
}

const emit = defineEmits<{
  update: [payload: UpdatePayload]
}>()

const handleChange = () => {
  emit('update', {
    field: 'name',
    value: 'John'
  })
}

// Component B - Receiving custom events
const handleUpdate = (payload: UpdatePayload) => {
  console.log(`Field ${payload.field} changed to:`, payload.value)
}
```

```vue
<template>
  <ComponentA @update="handleUpdate" />
</template>
```

## Utility Types for Vue3

### Common Utility Types

```typescript
import { Ref, ComputedRef, UnwrapRef } from 'vue'

// ✅ Unwrap nested refs
type UnwrappedUser = UnwrapRef<Ref<User>>

// ✅ Extract ref value type
type UserType = Ref<User> extends Ref<infer T> ? T : never

// ✅ Make properties optional
type PartialUser = Partial<User>

// ✅ Make properties required
type RequiredUser = Required<Partial<User>>

// ✅ Pick specific properties
type UserCredentials = Pick<User, 'email' | 'password'>

// ✅ Omit specific properties
type UserWithoutPassword = Omit<User, 'password'>

// ✅ Readonly
type ReadonlyUser = Readonly<User>
```

### Custom Utility Types

```typescript
// ✅ Make specific properties optional
type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

interface User {
  id: number
  name: string
  email: string
  avatar: string
}

type UserInput = PartialBy<User, 'id' | 'avatar'>
// { id?: number; name: string; email: string; avatar?: string }

// ✅ Deep partial
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

// ✅ Nullable
type Nullable<T> = T | null

// ✅ Maybe
type Maybe<T> = T | null | undefined
```

## Best Practices Summary

1. **Always use TypeScript in strict mode** - Enable `strict: true` in tsconfig.json
2. **Prefer type inference** - Let TypeScript infer types when obvious
3. **Use interfaces over types for objects** - Better error messages and extensibility
4. **Export types from composables** - Makes them reusable and testable
5. **Use const assertions** - For literal types: `const status = 'pending' as const`
6. **Avoid `any`** - Use `unknown` or proper types instead
7. **Use type guards** - For runtime type checking and narrowing
8. **Type your events** - Both emits and handlers
9. **Use generic types** - For reusable components and composables
10. **Keep types DRY** - Reuse and compose types, don't duplicate
