<template>
  <div :class="['relative w-full', $attrs.class]">
    <input
      :value="inputText"
      type="text"
      :placeholder="placeholder"
      @input="handleInput"
      @keydown="handleKeyDown"
      @focus="handleFocus"
      @blur="handleBlur"
      :class="[
        'w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200',
        inputClass
      ]"
    />
    <div
      v-if="showDropdown && filteredOptions.length > 0"
      class="absolute z-20 w-full mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-xl max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600"
    >
      <div
        v-for="(option, index) in filteredOptions"
        :key="index"
        @mousedown.prevent="selectOption(option)"
        :class="[
          'px-4 py-2 cursor-pointer text-sm text-gray-200 hover:bg-gray-700 transition-colors duration-150',
          index === selectedIndex ? 'bg-gray-700 text-white' : ''
        ]"
        v-html="highlight(option.text)"
      ></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { highlightText, splitIntoWords, normalizeText } from './autoCompleteUtils'

export interface AutoCompleteOption {
  text: string
  value: string | number
}

interface Props {
  modelValue: string | number
  options: (string | AutoCompleteOption)[]
  inputClass?: string
  placeholder?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: '',
  options: () => [],
  inputClass: '',
  placeholder: ''
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number]
  'change': [option: string | AutoCompleteOption]
}>()

const showDropdown = ref(false)
const selectedIndex = ref(-1)
const inputText = ref('')
let blurTimeout: number | null = null

const normalizedOptions = computed(() => {
  return props.options.map(opt => {
    if (typeof opt === 'string') {
      return { text: opt, value: opt }
    }
    return opt
  })
})

const findOptionByValue = (value: string | number): AutoCompleteOption | undefined => {
  return normalizedOptions.value.find(option => option.value === value)
}

const highlight = (text: string) => {
  return highlightText(inputText.value.trim(), text)
}

watch(() => props.modelValue, (newValue) => {
  const option = findOptionByValue(newValue)
  inputText.value = option ? option.text : String(newValue)
}, { immediate: true })

const filteredOptions = computed(() => {
  const text = inputText.value.trim()
  if (!text) {
    return normalizedOptions.value
  }
  
  const searchNormalized = normalizeText(text)
  const searchWords = splitIntoWords(text)
  
  return normalizedOptions.value.filter(option => {
    const textNormalized = normalizeText(option.text)
    
    if (textNormalized.includes(searchNormalized) || 
        option.text.toLowerCase().includes(text.toLowerCase())) {
      return true
    }
    
    if (searchWords.length > 1) {
      const textHasSeparator = /[_\s-]/.test(option.text)
      if (!textHasSeparator) {
        return searchWords.every(word => textNormalized.includes(normalizeText(word)))
      }
    }
    
    return false
  })
})

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  inputText.value = target.value
  selectedIndex.value = -1
  
  const hasInput = inputText.value.trim().length > 0
  showDropdown.value = hasInput && filteredOptions.value.length > 0
  
  // AutoComplete 允許自由輸入，直接更新 modelValue
  emit('update:modelValue', target.value)
}

const handleFocus = () => {
  if (blurTimeout) {
    clearTimeout(blurTimeout)
    blurTimeout = null
  }
  if (inputText.value.trim().length > 0 && filteredOptions.value.length > 0) {
    showDropdown.value = true
  }
}

const handleBlur = () => {
  blurTimeout = window.setTimeout(() => {
    showDropdown.value = false
    selectedIndex.value = -1
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
}

const selectOption = (option: AutoCompleteOption) => {
  inputText.value = option.text
  emit('update:modelValue', option.value)
  emit('change', option)
  showDropdown.value = false
  selectedIndex.value = -1
}
</script>
