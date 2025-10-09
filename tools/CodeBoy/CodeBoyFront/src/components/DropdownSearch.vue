<script setup lang="ts">
import { ref, computed } from 'vue'

export interface SearchItem {
  name: string
  description: string
  [key: string]: any
}

interface Props {
  items: SearchItem[]
  placeholder?: string
  selectedValue?: any
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: '搜尋功能...'
})

const emit = defineEmits<{
  select: [item: SearchItem]
}>()

const searchQuery = ref('')
const isSearchFocused = ref(false)

const filteredItems = computed(() => {
  if (!searchQuery.value.trim()) {
    return props.items
  }
  const query = searchQuery.value.toLowerCase()
  return props.items.filter(item => 
    item.name.toLowerCase().includes(query) || 
    item.description.toLowerCase().includes(query)
  )
})

const showDropdown = computed(() => {
  return isSearchFocused.value
})

const handleSearchFocus = () => {
  isSearchFocused.value = true
}

const handleSearchBlur = () => {
  setTimeout(() => {
    isSearchFocused.value = false
  }, 200)
}

const handleItemClick = (item: SearchItem) => {
  emit('select', item)
  searchQuery.value = ''
  isSearchFocused.value = false
}

const isItemSelected = (item: SearchItem) => {
  return props.selectedValue !== undefined && item === props.selectedValue
}
</script>

<template>
  <div class="relative flex-1 max-w-md">
    <div class="relative">
      <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
        </svg>
      </div>
      <input
        v-model="searchQuery"
        type="text"
        :placeholder="placeholder"
        class="w-full pl-10 pr-3 py-2 border border-gray-600 rounded-md bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        @focus="handleSearchFocus"
        @blur="handleSearchBlur"
      />
    </div>

    <div
      v-if="showDropdown"
      class="absolute z-50 mt-2 w-full bg-gray-800 border border-gray-600 rounded-md shadow-lg max-h-96 overflow-y-auto"
    >
      <div v-if="filteredItems.length === 0" class="px-4 py-3 text-sm text-gray-400">
        找不到符合的功能
      </div>
      <div v-else>
        <button
          v-for="(item, index) in filteredItems"
          :key="index"
          @click="handleItemClick(item)"
          class="w-full text-left px-4 py-3 hover:bg-gray-700 transition-colors border-b border-gray-700 last:border-b-0"
          :class="isItemSelected(item) ? 'bg-blue-900/30' : ''"
        >
          <div class="flex items-center justify-between">
            <div>
              <div class="text-sm font-medium text-white">{{ item.name }}</div>
              <div class="text-xs text-gray-400 mt-1">{{ item.description }}</div>
            </div>
            <svg
              v-if="isItemSelected(item)"
              class="h-5 w-5 text-blue-400 flex-shrink-0 ml-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
            </svg>
          </div>
        </button>
      </div>
    </div>
  </div>
</template>

