<template>
  <div :class="['relative w-full', $attrs.class]">
    <div class="relative">
      <input
        :value="inputText"
        type="text"
        :placeholder="placeholder"
        @input="handleInput"
        @keydown="handleKeyDown"
        @focus="handleFocus"
        @blur="handleBlur"
        :class="[
          'w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 pr-10',
          inputClass
        ]"
      />
      <div class="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
        <svg class="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </div>
    
    <div
      v-if="showDropdown && filteredOptions.length > 0"
      class="absolute z-20 w-full mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-xl max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600"
    >
      <div
        v-for="(option, index) in filteredOptions"
        :key="index"
        @mousedown.prevent="selectOption(option)"
        :class="[
          'px-4 py-2 cursor-pointer text-sm text-gray-200 hover:bg-gray-700 transition-colors duration-150 border-b border-gray-700 last:border-0',
          index === selectedIndex ? 'bg-gray-700 text-white' : '',
          props.modelValue === option.value ? 'text-blue-400 font-semibold' : ''
        ]"
        v-html="highlight(option.label)"
      ></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { highlightText, normalizeText, splitIntoWords } from './autoCompleteUtils'

export interface DropDownOption {
  label: string
  value: string | number
}

interface Props {
  modelValue?: string | number
  options: DropDownOption[]
  inputClass?: string
  placeholder?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: '',
  options: () => [],
  inputClass: '',
  placeholder: '請選擇...'
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number]
}>()

const showDropdown = ref(false)
const selectedIndex = ref(-1)
const inputText = ref('')
const originalLabel = ref('')
let blurTimeout: number | null = null

const findOptionByValue = (value?: string | number): DropDownOption | undefined => {
  return props.options.find(opt => opt.value === value)
}

const highlight = (text: string) => {
  return highlightText(inputText.value.trim(), text)
}

watch(() => props.modelValue, (newValue) => {
  const option = findOptionByValue(newValue)
  if (option) {
    inputText.value = option.label
    originalLabel.value = option.label
  } else {
    inputText.value = ''
    originalLabel.value = ''
  }
}, { immediate: true })

const filteredOptions = computed(() => {
  const text = inputText.value.trim()
  if (!text || text === originalLabel.value) {
    return props.options
  }
  
  const searchNormalized = normalizeText(text)
  const searchWords = splitIntoWords(text)
  
  return props.options.filter(option => {
    const textNormalized = normalizeText(option.label)
    
    if (textNormalized.includes(searchNormalized) || 
        option.label.toLowerCase().includes(text.toLowerCase())) {
      return true
    }
    
    if (searchWords.length > 1) {
      return searchWords.every(word => textNormalized.includes(normalizeText(word)))
    }
    
    return false
  })
})

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  inputText.value = target.value
  selectedIndex.value = -1
  showDropdown.value = true
}

const handleFocus = () => {
  if (blurTimeout) {
    clearTimeout(blurTimeout)
    blurTimeout = null
  }
  showDropdown.value = true
}

const handleBlur = () => {
  blurTimeout = window.setTimeout(() => {
    showDropdown.value = false
    selectedIndex.value = -1
    
    // 如果沒有選中任何東西，且輸入內容不符合任何選項，則還原
    const currentOption = findOptionByValue(props.modelValue)
    if (currentOption) {
      inputText.value = currentOption.label
    } else {
      inputText.value = ''
    }
  }, 200)
}

const handleKeyDown = (event: KeyboardEvent) => {
  if (event.key === 'Enter') {
    handleEnterKey()
  } else if (event.key === 'ArrowDown') {
    handleArrowDown()
  } else if (event.key === 'ArrowUp') {
    handleArrowUp()
  } else if (event.key === 'Escape') {
    handleEscapeKey()
  }
}

const handleEnterKey = () => {
  if (showDropdown.value && filteredOptions.value.length > 0) {
    if (selectedIndex.value >= 0 && selectedIndex.value < filteredOptions.value.length) {
      selectOption(filteredOptions.value[selectedIndex.value])
    } else if (filteredOptions.value.length === 1) {
      selectOption(filteredOptions.value[0])
    }
  }
}

const handleArrowDown = () => {
  if (!showDropdown.value) {
    showDropdown.value = true
    return
  }
  selectedIndex.value = (selectedIndex.value + 1) % filteredOptions.value.length
}

const handleArrowUp = () => {
  if (!showDropdown.value) return
  selectedIndex.value = selectedIndex.value <= 0
    ? filteredOptions.value.length - 1
    : selectedIndex.value - 1
}

const handleEscapeKey = () => {
  showDropdown.value = false
  selectedIndex.value = -1
  const currentOption = findOptionByValue(props.modelValue)
  if (currentOption) {
    inputText.value = currentOption.label
  }
}

const selectOption = (option: DropDownOption) => {
  inputText.value = option.label
  originalLabel.value = option.label
  emit('update:modelValue', option.value)
  showDropdown.value = false
  selectedIndex.value = -1
}
</script>
