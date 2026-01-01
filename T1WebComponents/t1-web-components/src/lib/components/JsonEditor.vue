<template>
  <div class="flex flex-col w-full bg-gray-900 border border-gray-700 rounded-lg overflow-hidden shadow-xl text-gray-200">
    <!-- Error Message -->
    <div
      v-if="errorMessage"
      class="m-4 p-3 bg-red-900/20 border border-red-700 rounded-md text-red-400 text-sm"
    >
      {{ errorMessage }}
      <div class="mt-2 p-2 bg-gray-900 rounded font-mono text-xs overflow-x-auto">
        {{ modelValue }}
      </div>
    </div>

    <!-- Header with Search and Add (Array Mode Only) -->
    <div v-if="isArray" class="bg-gray-800/50 border-b border-gray-700">
      <!-- Search and Add -->
      <div class="flex flex-col sm:flex-row p-4 gap-4 items-center justify-between">
        <div class="relative w-full sm:max-w-xs">
          <input
            v-model="searchQuery"
            type="text"
            placeholder="Search items..."
            class="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all"
          />
          <span class="absolute left-3 top-2.5 text-gray-500">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </span>
        </div>
        <div class="flex items-center gap-2">
          <button
            v-if="tempArrayData && tempArrayData.length > 0"
            @click="deleteAll"
            class="flex items-center gap-2 px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-900/50 rounded-md text-sm font-medium transition-all"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Delete All
          </button>
          <button
            @click="openAddModal"
            class="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors shadow-sm"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
            Add Item
          </button>
        </div>
      </div>

      <!-- Save/Cancel Buttons -->
      <div class="px-4 pb-4 flex items-center gap-3 justify-end">
        <button
          @click="cancelArrayChanges"
          :disabled="!hasUnsavedChanges"
          class="px-4 py-2 rounded-md border transition-colors"
          :class="hasUnsavedChanges
            ? 'border-gray-600 text-gray-300 hover:bg-gray-700'
            : 'border-gray-700 text-gray-600 cursor-not-allowed'"
        >
          Cancel
        </button>
        <button
          @click="saveArrayChanges"
          :disabled="!hasUnsavedChanges"
          class="px-4 py-2 rounded-md font-medium transition-colors"
          :class="hasUnsavedChanges
            ? 'bg-purple-500 text-white hover:bg-purple-600'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'"
        >
          Save
        </button>
      </div>
    </div>

    <!-- Table Body (Array Mode) -->
    <div v-if="isArray" class="overflow-x-auto">
      <table class="w-full text-left text-sm border-collapse">
        <thead class="bg-gray-800 text-gray-400 font-medium uppercase tracking-wider border-b border-gray-700">
          <tr>
            <th v-for="field in effectiveSchema" :key="field.key" class="px-6 py-3">
              {{ field.label || field.key }}
            </th>
            <th class="px-6 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-800">
          <tr v-for="(item, index) in filteredList" :key="index" class="hover:bg-gray-800/30 transition-colors">
            <td v-for="field in effectiveSchema" :key="field.key" class="px-6 py-4 whitespace-nowrap">
              <span v-if="field.type === 'date'" class="text-gray-300">
                {{ formatValue(item[field.key]) }}
              </span>
              <span v-else class="text-gray-300">
                {{ item[field.key] }}
              </span>
            </td>
            <td class="px-6 py-4 text-right whitespace-nowrap space-x-2">
              <button @click="openInsertModal(item)" class="text-green-400 hover:text-green-300 transition-colors">
                Add Before
              </button>
              <button @click="openEditModal(item)" class="text-blue-400 hover:text-blue-300 transition-colors">
                Edit
              </button>
              <button @click="deleteItem(item)" class="text-red-400 hover:text-red-300 transition-colors">
                Delete
              </button>
            </td>
          </tr>
          <tr v-if="filteredList.length === 0">
            <td :colspan="effectiveSchema.length + 1" class="px-6 py-12 text-center text-gray-500 italic">
              No matching items found.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Form Mode (Object Mode) -->
    <div v-else class="p-6">
      <!-- Empty State -->
      <div
        v-if="!tempObjectData"
        class="p-8 text-center text-gray-500"
      >
        No data. Please provide a valid JSON object.
      </div>

      <!-- Form Fields -->
      <div v-else>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div v-for="field in effectiveSchema" :key="field.key" class="space-y-1">
            <label :for="'form-' + field.key" class="block text-xs font-medium text-gray-400 uppercase tracking-wider">
              {{ field.label || field.key }}
            </label>
            <input
              v-if="field.type === 'string'"
              :id="'form-' + field.key"
              v-model="tempObjectData[field.key]"
              @input="onObjectChange"
              class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:ring-2 focus:ring-blue-500 text-white text-sm"
              type="text"
            />
            <input
              v-else-if="field.type === 'number'"
              :id="'form-' + field.key"
              v-model.number="tempObjectData[field.key]"
              @input="onObjectChange"
              class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:ring-2 focus:ring-blue-500 text-white text-sm"
              type="number"
            />
            <input
              v-else-if="field.type === 'date'"
              :id="'form-' + field.key"
              v-model="tempObjectData[field.key]"
              @input="onObjectChange"
              class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:ring-2 focus:ring-blue-500 text-white text-sm color-scheme-dark"
              type="date"
            />
          </div>
        </div>

        <!-- Save/Cancel Buttons -->
        <div class="mt-6 flex items-center gap-3 justify-end">
          <button
            @click="cancelObjectChanges"
            :disabled="!hasUnsavedChanges"
            class="px-4 py-2 rounded-md border transition-colors"
            :class="hasUnsavedChanges
              ? 'border-gray-600 text-gray-300 hover:bg-gray-700'
              : 'border-gray-700 text-gray-600 cursor-not-allowed'"
          >
            Cancel
          </button>
          <button
            @click="saveObjectChanges"
            :disabled="!hasUnsavedChanges"
            class="px-4 py-2 rounded-md font-medium transition-colors"
            :class="hasUnsavedChanges
              ? 'bg-orange-500 text-white hover:bg-orange-600'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'"
          >
            Save
          </button>
        </div>
      </div>
    </div>

    <!-- Modal for Add/Edit -->
    <div v-if="isModalOpen" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div class="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-md overflow-hidden animate-in fade-in zoom-in duration-200">
        <div class="p-6 border-b border-gray-800 flex justify-between items-center bg-gray-800/30">
          <h3 class="text-lg font-semibold text-white">
            {{ editingIndex !== -1 ? 'Edit Item' : (insertIndex !== -1 ? 'Add Item Before' : 'Add New Item') }}
          </h3>
          <button @click="closeModal" class="text-gray-500 hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div class="p-6 space-y-4 max-h-[70vh] overflow-y-auto">
          <div v-for="field in effectiveSchema" :key="field.key" class="space-y-1">
            <label :for="field.key" class="block text-xs font-medium text-gray-400 uppercase tracking-wider">
              {{ field.label || field.key }}
            </label>
            <input
              v-if="field.type === 'string'"
              :id="field.key"
              v-model="tempItem[field.key]"
              class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:ring-2 focus:ring-blue-500 text-white text-sm"
              type="text"
            />
            <input
              v-else-if="field.type === 'number'"
              :id="field.key"
              v-model.number="tempItem[field.key]"
              class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:ring-2 focus:ring-blue-500 text-white text-sm"
              type="number"
            />
            <input
              v-else-if="field.type === 'date'"
              :id="field.key"
              v-model="tempItem[field.key]"
              class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:ring-2 focus:ring-blue-500 text-white text-sm color-scheme-dark"
              type="date"
            />
          </div>
        </div>

        <div class="p-6 bg-gray-800/30 border-t border-gray-800 flex justify-end gap-3">
          <button @click="closeModal" class="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors">
            Cancel
          </button>
          <button
            @click="saveItem"
            class="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold rounded-md shadow-lg transition-transform active:scale-95"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

export interface JsonSchemaField {
  key: string
  label?: string
  type: 'string' | 'number' | 'date'
}

const props = withDefaults(
  defineProps<{
    modelValue: string | null
    schema?: JsonSchemaField[] | null
    compact?: boolean
  }>(),
  {
    compact: false,
    schema: null
  }
)

const emit = defineEmits<{
  'update:modelValue': [value: string]
  'change': [value: string]
  'error': [message: string]
}>()

const internalData = ref<any>(null)
const tempArrayData = ref<any[]>([])
const tempObjectData = ref<any>(null)
const errorMessage = ref<string>('')
const hasUnsavedChanges = ref<boolean>(false)

const effectiveSchema = computed(() => {
  if (props.schema && props.schema.length > 0) {
    return props.schema
  }

  if (props.modelValue) {
    const extracted = extractSchema(props.modelValue)
    if (Array.isArray(extracted)) {
      return extracted
    }
  }

  return []
})

const initializeData = () => {
  if (!props.modelValue || props.modelValue.trim() === '') {
    if (props.modelValue?.trim().startsWith('[')) {
      return []
    }
    return null
  }

  try {
    errorMessage.value = ''
    return JSON.parse(props.modelValue)
  } catch (error: any) {
    errorMessage.value = `Invalid JSON: ${error.message}`
    emit('error', `JSON parse failed: ${error.message}`)
    return null
  }
}

const isArray = computed(() => {
  if (!props.modelValue || props.modelValue.trim() === '') {
    return false
  }
  const trimmed = props.modelValue.trim()
  return trimmed.startsWith('[')
})

const initializeObjectFromSchema = () => {
  const obj: any = {}
  effectiveSchema.value.forEach(field => {
    obj[field.key] = field.type === 'number' ? 0 : ''
  })
  return obj
}

watch(
  () => props.modelValue,
  () => {
    internalData.value = initializeData()

    if (isArray.value) {
      if (Array.isArray(internalData.value)) {
        tempArrayData.value = JSON.parse(JSON.stringify(internalData.value))
      } else {
        tempArrayData.value = []
      }
      hasUnsavedChanges.value = false
    } else {
      if (internalData.value) {
        tempObjectData.value = JSON.parse(JSON.stringify(internalData.value))
      } else {
        tempObjectData.value = initializeObjectFromSchema()
      }
      hasUnsavedChanges.value = false
    }
  },
  { immediate: true }
)

const searchQuery = ref('')
const isModalOpen = ref(false)
const editingIndex = ref(-1)
const insertIndex = ref(-1)
const tempItem = ref<any>({})

const filteredList = computed(() => {
  if (!isArray.value || !tempArrayData.value) return []
  if (!searchQuery.value) return tempArrayData.value

  const q = searchQuery.value.toLowerCase()
  return tempArrayData.value.filter((item: any) => {
    return Object.values(item).some(val =>
      String(val).toLowerCase().includes(q)
    )
  })
})

const formatValue = (val: any) => {
  if (!val) return ''
  return String(val)
}

const openAddModal = () => {
  editingIndex.value = -1
  insertIndex.value = -1
  initializeTempItem()
  isModalOpen.value = true
}

const openInsertModal = (item: any) => {
  editingIndex.value = -1
  insertIndex.value = tempArrayData.value.indexOf(item)
  initializeTempItem()
  isModalOpen.value = true
}

const initializeTempItem = () => {
  const newItem: any = {}
  effectiveSchema.value.forEach(field => {
    newItem[field.key] = field.type === 'number' ? 0 : ''
  })
  tempItem.value = newItem
}

const openEditModal = (item: any) => {
  editingIndex.value = tempArrayData.value.indexOf(item)
  insertIndex.value = -1
  tempItem.value = JSON.parse(JSON.stringify(item))
  isModalOpen.value = true
}

const closeModal = () => {
  isModalOpen.value = false
  tempItem.value = {}
  editingIndex.value = -1
  insertIndex.value = -1
}

const saveItem = () => {
  const newList = [...tempArrayData.value]
  if (editingIndex.value !== -1) {
    newList[editingIndex.value] = { ...tempItem.value }
  } else if (insertIndex.value !== -1) {
    newList.splice(insertIndex.value, 0, { ...tempItem.value })
  } else {
    newList.push({ ...tempItem.value })
  }

  tempArrayData.value = newList
  hasUnsavedChanges.value = true
  closeModal()
}

const deleteItem = (item: any) => {
  const actualIndex = tempArrayData.value.indexOf(item)
  if (actualIndex > -1 && confirm('Are you sure you want to delete this item?')) {
    const newList = [...tempArrayData.value]
    newList.splice(actualIndex, 1)
    tempArrayData.value = newList
    hasUnsavedChanges.value = true
  }
}

const deleteAll = () => {
  if (confirm('Are you sure you want to delete ALL items? This action cannot be undone.')) {
    tempArrayData.value = []
    hasUnsavedChanges.value = true
  }
}

const serializeAndEmit = (data: any) => {
  try {
    const jsonString = props.compact
      ? JSON.stringify(data)
      : JSON.stringify(data, null, 2)

    errorMessage.value = ''
    emit('update:modelValue', jsonString)
    emit('change', jsonString)
  } catch (error: any) {
    errorMessage.value = `Serialize failed: ${error.message}`
    emit('error', `JSON stringify failed: ${error.message}`)
  }
}

const onObjectChange = () => {
  hasUnsavedChanges.value = true
}

const saveObjectChanges = () => {
  if (!tempObjectData.value) return

  internalData.value = JSON.parse(JSON.stringify(tempObjectData.value))
  serializeAndEmit(internalData.value)
  hasUnsavedChanges.value = false
}

const cancelObjectChanges = () => {
  if (!internalData.value) return

  tempObjectData.value = JSON.parse(JSON.stringify(internalData.value))
  hasUnsavedChanges.value = false
}

const saveArrayChanges = () => {
  if (!tempArrayData.value) return

  internalData.value = JSON.parse(JSON.stringify(tempArrayData.value))
  serializeAndEmit(internalData.value)
  hasUnsavedChanges.value = false
}

const cancelArrayChanges = () => {
  if (!internalData.value) return

  tempArrayData.value = JSON.parse(JSON.stringify(internalData.value))
  hasUnsavedChanges.value = false
}

function inferFieldType(value: any): 'string' | 'number' | 'date' {
  if (typeof value === 'number') return 'number'

  if (typeof value === 'string') {
    const isoDateRegex = /^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$/
    if (isoDateRegex.test(value)) {
      const date = new Date(value)
      if (!isNaN(date.getTime())) return 'date'
    }
  }

  return 'string'
}

function extractSchema(jsonStr: string | null): JsonSchemaField[] | {} {
  if (!jsonStr || jsonStr.trim() === '') return {}

  let data: any
  try {
    data = JSON.parse(jsonStr)
  } catch {
    return {}
  }

  if (Array.isArray(data)) {
    if (data.length === 0) return []

    const allKeys = new Set<string>()
    const typeCountMap = new Map<string, Map<'string' | 'number' | 'date', number>>()

    data.forEach((item) => {
      if (typeof item === 'object' && item !== null) {
        Object.keys(item).forEach((key) => {
          allKeys.add(key)

          const value = item[key]
          if (value !== undefined && value !== null) {
            const type = inferFieldType(value)

            if (!typeCountMap.has(key)) {
              typeCountMap.set(key, new Map())
            }
            const counts = typeCountMap.get(key)!
            counts.set(type, (counts.get(type) || 0) + 1)
          }
        })
      }
    })

    return Array.from(allKeys).map((key) => {
      const counts = typeCountMap.get(key)
      let finalType: 'string' | 'number' | 'date' = 'string'

      if (counts) {
        let maxCount = 0
        counts.forEach((count, type) => {
          if (count > maxCount) {
            maxCount = count
            finalType = type
          }
        })
      }

      return { key, type: finalType }
    })
  }

  if (typeof data === 'object' && data !== null) {
    return Object.keys(data).map((key) => ({
      key,
      type: inferFieldType(data[key])
    }))
  }

  return {}
}

defineExpose({
  extractSchema,
  saveArrayChanges,
  cancelArrayChanges
})
</script>

<style scoped>
.color-scheme-dark {
  color-scheme: dark;
}
</style>
