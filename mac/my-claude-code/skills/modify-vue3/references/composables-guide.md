# Composables Guide

## What is a Composable?

A composable is a function that leverages Vue's Composition API to encapsulate and reuse **stateful logic**. Composables are the Vue3 equivalent of mixins or React hooks.

### Key Characteristics

- **Prefix with `use`**: Convention for identifying composables
- **Stateful**: Manages reactive state using Vue's reactivity system
- **Reusable**: Can be imported and used in multiple components
- **Composable**: Can be combined with other composables
- **Return reactive state and functions**: Provides both data and methods

## When to Create a Composable

### ✅ Create a composable when you have:

1. **Repeated logic across components**
   - Same data fetching pattern in multiple places
   - Similar form handling logic
   - Common UI interaction patterns (modals, dropdowns, etc.)

2. **Complex stateful logic**
   - Logic that needs reactive state management
   - Side effects that need lifecycle management
   - Watchers or computed properties

3. **Cross-cutting concerns**
   - Authentication state
   - Window resize handling
   - Network status
   - Theme management

### ❌ Don't create a composable for:

1. **Simple utility functions** without state
   ```typescript
   // ❌ Don't make this a composable
   export function useFormatDate(date: Date) {
     return date.toISOString()
   }
   
   // ✅ Just a utility function
   export function formatDate(date: Date) {
     return date.toISOString()
   }
   ```

2. **One-time use logic** that won't be reused

3. **Component-specific logic** that's too coupled to a single component

## Basic Structure

### Minimal Composable

```typescript
import { ref } from 'vue'

export function useCounter(initialValue = 0) {
  // State
  const count = ref(initialValue)
  
  // Methods
  const increment = () => count.value++
  const decrement = () => count.value--
  const reset = () => count.value = initialValue
  
  // Return public API
  return {
    count,
    increment,
    decrement,
    reset
  }
}
```

### With TypeScript

```typescript
import { ref, Ref, readonly } from 'vue'

export interface UseCounterReturn {
  count: Readonly<Ref<number>>
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
    count: readonly(count), // Prevent external mutation
    increment,
    decrement,
    reset
  }
}
```

## Common Patterns

### Pattern 1: Data Fetching

```typescript
import { ref, Ref } from 'vue'

export interface UseApiReturn<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<Error | null>
  execute: () => Promise<void>
  reset: () => void
}

export function useApi<T>(url: string): UseApiReturn<T> {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)
  
  const execute = async () => {
    loading.value = true
    error.value = null
    
    try {
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      data.value = await response.json()
    } catch (e) {
      error.value = e as Error
      data.value = null
    } finally {
      loading.value = false
    }
  }
  
  const reset = () => {
    data.value = null
    loading.value = false
    error.value = null
  }
  
  return { data, loading, error, execute, reset }
}

// Usage
const { data, loading, error, execute } = useApi<User[]>('/api/users')

onMounted(() => {
  execute()
})
```

### Pattern 2: Form Management

```typescript
import { reactive, computed } from 'vue'

export interface UseFormReturn<T> {
  formData: T
  errors: Record<string, string>
  isValid: Ref<boolean>
  isDirty: Ref<boolean>
  validate: () => boolean
  reset: () => void
  setField: (field: keyof T, value: any) => void
}

export function useForm<T extends Record<string, any>>(
  initialData: T,
  validationRules?: Partial<Record<keyof T, (value: any) => string | null>>
): UseFormReturn<T> {
  const formData = reactive<T>({ ...initialData })
  const errors = reactive<Record<string, string>>({})
  const touched = reactive<Record<string, boolean>>({})
  
  const isValid = computed(() => Object.keys(errors).length === 0)
  const isDirty = computed(() => 
    Object.keys(touched).some(key => touched[key])
  )
  
  const validate = () => {
    Object.keys(errors).forEach(key => delete errors[key])
    
    if (!validationRules) return true
    
    let valid = true
    Object.keys(validationRules).forEach((field) => {
      const rule = validationRules[field as keyof T]
      if (rule) {
        const error = rule(formData[field])
        if (error) {
          errors[field] = error
          valid = false
        }
      }
    })
    
    return valid
  }
  
  const reset = () => {
    Object.assign(formData, initialData)
    Object.keys(errors).forEach(key => delete errors[key])
    Object.keys(touched).forEach(key => delete touched[key])
  }
  
  const setField = (field: keyof T, value: any) => {
    formData[field] = value
    touched[field as string] = true
    
    // Validate single field
    if (validationRules && validationRules[field]) {
      const error = validationRules[field]!(value)
      if (error) {
        errors[field as string] = error
      } else {
        delete errors[field as string]
      }
    }
  }
  
  return {
    formData,
    errors,
    isValid,
    isDirty,
    validate,
    reset,
    setField
  }
}

// Usage
const { formData, errors, isValid, validate, reset } = useForm(
  { name: '', email: '' },
  {
    name: (value) => !value ? 'Name is required' : null,
    email: (value) => {
      if (!value) return 'Email is required'
      if (!/\S+@\S+\.\S+/.test(value)) return 'Invalid email'
      return null
    }
  }
)
```

### Pattern 3: Modal/Toggle State

```typescript
import { ref, Ref } from 'vue'

export interface UseToggleReturn {
  isOpen: Ref<boolean>
  open: () => void
  close: () => void
  toggle: () => void
}

export function useToggle(initialState = false): UseToggleReturn {
  const isOpen = ref(initialState)
  
  const open = () => isOpen.value = true
  const close = () => isOpen.value = false
  const toggle = () => isOpen.value = !isOpen.value
  
  return { isOpen, open, close, toggle }
}

// Usage
const { isOpen, open, close, toggle } = useToggle()
```

### Pattern 4: Async State with Auto-execution

```typescript
import { ref, onMounted } from 'vue'

export function useAsyncData<T>(
  fetcher: () => Promise<T>,
  options: { immediate?: boolean } = { immediate: true }
) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)
  
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
  
  if (options.immediate) {
    onMounted(execute)
  }
  
  return { data, loading, error, execute }
}

// Usage - auto-executes on mount
const { data, loading, error } = useAsyncData(() => 
  fetch('/api/users').then(r => r.json())
)

// Usage - manual execution
const { data, loading, error, execute } = useAsyncData(
  () => fetch('/api/users').then(r => r.json()),
  { immediate: false }
)
```

### Pattern 5: Event Listeners with Cleanup

```typescript
import { onMounted, onUnmounted, Ref } from 'vue'

export function useEventListener(
  target: Ref<EventTarget | null> | EventTarget,
  event: string,
  handler: EventListener,
  options?: AddEventListenerOptions
) {
  const cleanup = () => {
    const element = 'value' in target ? target.value : target
    element?.removeEventListener(event, handler, options)
  }
  
  const setup = () => {
    const element = 'value' in target ? target.value : target
    element?.addEventListener(event, handler, options)
  }
  
  onMounted(setup)
  onUnmounted(cleanup)
  
  return cleanup
}

// Usage with window
useEventListener(window, 'resize', () => {
  console.log('Window resized')
})

// Usage with ref
const buttonRef = ref<HTMLButtonElement | null>(null)
useEventListener(buttonRef, 'click', () => {
  console.log('Button clicked')
})
```

### Pattern 6: Debounced Value

```typescript
import { ref, watch, Ref } from 'vue'

export function useDebounce<T>(
  value: Ref<T>,
  delay = 300
): Ref<T> {
  const debouncedValue = ref<T>(value.value) as Ref<T>
  let timeoutId: number | undefined
  
  watch(value, (newValue) => {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    
    timeoutId = setTimeout(() => {
      debouncedValue.value = newValue
    }, delay) as unknown as number
  })
  
  return debouncedValue
}

// Usage
const searchQuery = ref('')
const debouncedQuery = useDebounce(searchQuery, 500)

watch(debouncedQuery, (query) => {
  // This will only execute 500ms after user stops typing
  performSearch(query)
})
```

### Pattern 7: Local Storage Sync

```typescript
import { ref, watch, Ref } from 'vue'

export function useLocalStorage<T>(
  key: string,
  defaultValue: T
): Ref<T> {
  const data = ref<T>(defaultValue) as Ref<T>
  
  // Load from localStorage on mount
  try {
    const stored = localStorage.getItem(key)
    if (stored) {
      data.value = JSON.parse(stored)
    }
  } catch (e) {
    console.error('Failed to parse localStorage value:', e)
  }
  
  // Save to localStorage on change
  watch(data, (newValue) => {
    try {
      localStorage.setItem(key, JSON.stringify(newValue))
    } catch (e) {
      console.error('Failed to save to localStorage:', e)
    }
  }, { deep: true })
  
  return data
}

// Usage
const theme = useLocalStorage('theme', 'light')
// Automatically synced with localStorage
```

## Advanced Patterns

### Composable with Dependencies

```typescript
// Composable that uses another composable
export function useUserProfile(userId: Ref<number>) {
  const { data: user, loading, error, execute } = useApi<User>(
    computed(() => `/api/users/${userId.value}`)
  )
  
  const fullName = computed(() => 
    user.value ? `${user.value.firstName} ${user.value.lastName}` : ''
  )
  
  const isAdmin = computed(() => 
    user.value?.role === 'admin'
  )
  
  // Re-fetch when userId changes
  watch(userId, () => {
    execute()
  }, { immediate: true })
  
  return {
    user,
    loading,
    error,
    fullName,
    isAdmin,
    refetch: execute
  }
}
```

### Generic Reusable Composables

```typescript
export function usePagination<T>(
  items: Ref<T[]>,
  itemsPerPage = 10
) {
  const currentPage = ref(1)
  
  const totalPages = computed(() => 
    Math.ceil(items.value.length / itemsPerPage)
  )
  
  const paginatedItems = computed(() => {
    const start = (currentPage.value - 1) * itemsPerPage
    const end = start + itemsPerPage
    return items.value.slice(start, end)
  })
  
  const nextPage = () => {
    if (currentPage.value < totalPages.value) {
      currentPage.value++
    }
  }
  
  const prevPage = () => {
    if (currentPage.value > 1) {
      currentPage.value--
    }
  }
  
  const goToPage = (page: number) => {
    if (page >= 1 && page <= totalPages.value) {
      currentPage.value = page
    }
  }
  
  return {
    currentPage,
    totalPages,
    paginatedItems,
    nextPage,
    prevPage,
    goToPage
  }
}

// Usage
const allItems = ref<Product[]>([/* ... */])
const { paginatedItems, currentPage, totalPages, nextPage, prevPage } = 
  usePagination(allItems, 20)
```

### Composable with Options Pattern

```typescript
interface UseFetchOptions<T> {
  immediate?: boolean
  refetch?: number // Interval in ms
  onSuccess?: (data: T) => void
  onError?: (error: Error) => void
  transform?: (data: any) => T
}

export function useFetch<T>(
  url: Ref<string> | string,
  options: UseFetchOptions<T> = {}
) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)
  
  const urlRef = isRef(url) ? url : ref(url)
  let intervalId: number | undefined
  
  const execute = async () => {
    loading.value = true
    error.value = null
    
    try {
      const response = await fetch(urlRef.value)
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
  
  // Auto-fetch on mount
  if (options.immediate !== false) {
    onMounted(execute)
  }
  
  // Auto-refetch on interval
  if (options.refetch) {
    intervalId = setInterval(execute, options.refetch) as unknown as number
    onUnmounted(() => {
      if (intervalId) clearInterval(intervalId)
    })
  }
  
  // Re-fetch when URL changes
  watch(urlRef, execute)
  
  return { data, loading, error, execute }
}
```

## Best Practices

### 1. Name with `use` Prefix

```typescript
// ✅ Good
export function useCounter() { }
export function useApi() { }
export function useForm() { }

// ❌ Bad
export function counter() { }
export function fetchData() { }
```

### 2. Return Consistent Structure

```typescript
// ✅ Good - return object with named properties
export function useData() {
  return {
    data,
    loading,
    error,
    refetch
  }
}

// ❌ Avoid - returning array (less clear)
export function useData() {
  return [data, loading, error, refetch]
}
```

### 3. Use `readonly` for State You Don't Want Mutated

```typescript
export function useCounter() {
  const count = ref(0)
  
  return {
    count: readonly(count), // Prevent external mutation
    increment: () => count.value++
  }
}
```

### 4. Export TypeScript Types

```typescript
export interface UseCounterReturn {
  count: Readonly<Ref<number>>
  increment: () => void
}

export function useCounter(): UseCounterReturn {
  // Implementation
}
```

### 5. Handle Cleanup Properly

```typescript
export function useInterval(callback: () => void, delay: number) {
  let intervalId: number | undefined
  
  const start = () => {
    if (intervalId) return
    intervalId = setInterval(callback, delay) as unknown as number
  }
  
  const stop = () => {
    if (intervalId) {
      clearInterval(intervalId)
      intervalId = undefined
    }
  }
  
  // Auto-cleanup on unmount
  onUnmounted(stop)
  
  return { start, stop }
}
```

### 6. Provide Sensible Defaults

```typescript
export function useApi<T>(
  url: string,
  options: {
    immediate?: boolean
    method?: string
  } = {
    immediate: true,
    method: 'GET'
  }
) {
  // Implementation
}
```

### 7. Make It Testable

```typescript
// ✅ Testable - dependency injection
export function useAuth(api: ApiClient) {
  // Can inject mock API for testing
}

// ✅ Testable - pure functions
export function useCalculator(initial: number) {
  // Easy to test with different initial values
}
```

## Testing Composables

```typescript
import { describe, it, expect } from 'vitest'
import { useCounter } from './useCounter'

describe('useCounter', () => {
  it('initializes with default value', () => {
    const { count } = useCounter()
    expect(count.value).toBe(0)
  })
  
  it('initializes with custom value', () => {
    const { count } = useCounter(10)
    expect(count.value).toBe(10)
  })
  
  it('increments count', () => {
    const { count, increment } = useCounter()
    increment()
    expect(count.value).toBe(1)
  })
  
  it('resets to initial value', () => {
    const { count, increment, reset } = useCounter(5)
    increment()
    increment()
    expect(count.value).toBe(7)
    reset()
    expect(count.value).toBe(5)
  })
})
```

## Common Mistakes to Avoid

### ❌ Calling composables conditionally

```typescript
// ❌ Wrong
if (someCondition) {
  const { data } = useApi('/api/data')
}

// ✅ Correct
const { data } = useApi('/api/data')
if (someCondition) {
  // Use data
}
```

### ❌ Calling composables in loops

```typescript
// ❌ Wrong
items.forEach(item => {
  const { data } = useApi(`/api/${item.id}`)
})

// ✅ Correct - create a separate component
// or use a single composable with reactive URL
```

### ❌ Not handling cleanup

```typescript
// ❌ Wrong
export function useWebSocket(url: string) {
  const ws = new WebSocket(url)
  // Missing cleanup!
}

// ✅ Correct
export function useWebSocket(url: string) {
  const ws = new WebSocket(url)
  
  onUnmounted(() => {
    ws.close()
  })
}
```
