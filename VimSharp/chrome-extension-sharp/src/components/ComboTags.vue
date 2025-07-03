<script setup lang="ts">
import { ref, watch, computed } from 'vue';

interface ComboTagItem {
  label: string;
  value: string | number;
}

const props = defineProps<{
  list: ComboTagItem[];
  modelValue: ComboTagItem[];
}>();
const emit = defineEmits<{
  (e: 'update:modelValue', value: ComboTagItem[]): void;
  (e: 'change', value: ComboTagItem[]): void;
}>();

const value = ref<ComboTagItem[]>([...props.modelValue]);
const input = ref<string>('');
const showDropdown = ref<boolean>(false);
const highlight = ref<number>(0);

const filtered = computed<ComboTagItem[]>(() => {
  const inputVal = input.value.toLowerCase();
  return props.list.filter(
    item => item.label.toLowerCase().includes(inputVal) && !value.value.some(v => v.value === item.value)
  );
});

watch(() => props.modelValue, (val) => {
  value.value = [...val];
});

function onInput() {
  showDropdown.value = !!input.value;
  highlight.value = 0;
}
function onFocus() {
  showDropdown.value = true;
}
function onBlur() {
  setTimeout(() => {
    showDropdown.value = false;
  }, 150);
}
function onKeyDown(e: KeyboardEvent) {
  if (!showDropdown.value) return;
  if (e.key === 'ArrowDown') {
    highlight.value = (highlight.value + 1) % filtered.value.length;
    e.preventDefault();
  } else if (e.key === 'ArrowUp') {
    highlight.value = (highlight.value - 1 + filtered.value.length) % filtered.value.length;
    e.preventDefault();
  } else if (e.key === 'Enter') {
    if (filtered.value[highlight.value]) {
      select(filtered.value[highlight.value]);
    }
    e.preventDefault();
  }
}
function select(item: ComboTagItem) {
  value.value = [...value.value, item];
  emit('update:modelValue', value.value);
  emit('change', value.value);
  input.value = '';
  showDropdown.value = false;
  highlight.value = 0;
}
function removeTag(idx: number) {
  value.value = value.value.filter((_, i) => i !== idx);
  emit('update:modelValue', value.value);
  emit('change', value.value);
}
</script>


<template>
  <div class="combo-wrap">
    <div class="tags-wrap">
      <span
        v-for="(tag, i) in value"
        :key="tag.value"
        class="tag"
      >
        {{ tag.label }}<span class="tag-x" @click="removeTag(i)">&times;</span>
      </span>
      <input
        class="textbox"
        type="text"
        v-model="input"
        placeholder="Please input keyword..."
        @input="onInput"
        @focus="onFocus"
        @blur="onBlur"
        @keydown="onKeyDown"
        :aria-expanded="showDropdown ? 'true' : undefined"
      />
    </div>
    <div v-if="showDropdown && filtered.length" class="dropdown">
      <div
        v-for="(item, idx) in filtered"
        :key="item.value"
        class="dropdown-item"
        :class="{ active: idx === highlight }"
        @mousedown.prevent="select(item)"
      >
        {{ item.label }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.tag {
  @apply inline-flex items-center bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full px-3 py-1 mr-2 mb-1 text-sm;
}
.tag-x {
  @apply ml-2 cursor-pointer text-blue-400 hover:text-red-500 dark:text-blue-300 dark:hover:text-red-400;
}
.dropdown {
  @apply absolute z-10 bg-white border border-gray-300 rounded shadow w-full mt-1 max-h-48 overflow-auto dark:bg-gray-800 dark:border-gray-600 dark:shadow-lg;
}
.dropdown-item {
  @apply px-4 py-2 cursor-pointer hover:bg-blue-100 dark:hover:bg-blue-900 dark:text-gray-100;
}
.dropdown-item.active {
  @apply bg-blue-200 dark:bg-blue-700;
}
.textbox {
  @apply border border-gray-300 rounded px-2 py-1 w-full focus:outline-none dark:bg-gray-900 dark:border-gray-600 dark:text-gray-100;
}
.tags-wrap {
  @apply flex flex-wrap items-center gap-1;
}
.combo-wrap {
  @apply relative w-full;
}
</style> 