<template>
  <div class="flex flex-col w-full bg-gray-900 border border-gray-700 rounded-lg overflow-hidden shadow-xl text-gray-200">
    <!-- Header with Search and Add -->
    <div class="flex flex-col sm:flex-row p-4 gap-4 bg-gray-800/50 border-b border-gray-700 items-center justify-between">
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

    <!-- Table Body -->
    <div class="overflow-x-auto">
      <table class="w-full text-left text-sm border-collapse">
        <thead class="bg-gray-800 text-gray-400 font-medium uppercase tracking-wider border-b border-gray-700">
          <tr>
            <th v-for="field in schema" :key="field.key" class="px-6 py-3">
              {{ field.label || field.key }}
            </th>
            <th class="px-6 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-800">
          <tr v-for="(item, index) in filteredList" :key="index" class="hover:bg-gray-800/30 transition-colors">
            <td v-for="field in schema" :key="field.key" class="px-6 py-4 whitespace-nowrap">
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
            <td :colspan="schema.length + 1" class="px-6 py-12 text-center text-gray-500 italic">
              No matching items found.
            </td>
          </tr>
        </tbody>
      </table>
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
          <div v-for="field in schema" :key="field.key" class="space-y-1">
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

const props = defineProps<{
  modelValue: any[]
  schema: JsonSchemaField[]
}>()

const emit = defineEmits(['update:modelValue', 'change'])

const searchQuery = ref('')
const isModalOpen = ref(false)
const editingIndex = ref(-1)
const insertIndex = ref(-1)
const tempItem = ref<any>({})

const filteredList = computed(() => {
  if (!searchQuery.value) return props.modelValue
  const q = searchQuery.value.toLowerCase()
  return props.modelValue.filter(item => {
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
  insertIndex.value = props.modelValue.indexOf(item)
  initializeTempItem()
  isModalOpen.value = true
}

const initializeTempItem = () => {
  const newItem: any = {}
  props.schema.forEach(field => {
    newItem[field.key] = field.type === 'number' ? 0 : ''
  })
  tempItem.value = newItem
}

const openEditModal = (item: any) => {
  editingIndex.value = props.modelValue.indexOf(item)
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
  const newList = [...props.modelValue]
  if (editingIndex.value !== -1) {
    newList[editingIndex.value] = { ...tempItem.value }
  } else if (insertIndex.value !== -1) {
    newList.splice(insertIndex.value, 0, { ...tempItem.value })
  } else {
    newList.push({ ...tempItem.value })
  }
  
  updateValue(newList)
  closeModal()
}

const deleteItem = (item: any) => {
  const actualIndex = props.modelValue.indexOf(item)
  if (actualIndex > -1 && confirm('Are you sure you want to delete this item?')) {
    const newList = [...props.modelValue]
    newList.splice(actualIndex, 1)
    updateValue(newList)
  }
}

const updateValue = (newList: any[]) => {
  emit('update:modelValue', newList)
  emit('change', newList)
}
</script>

<style scoped>
.color-scheme-dark {
  color-scheme: dark;
}
</style>
