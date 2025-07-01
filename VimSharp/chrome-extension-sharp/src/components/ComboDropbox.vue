<script lang="ts" setup>
import { ref, computed, watch, defineProps, defineEmits } from 'vue';

// 定義 props
const props = defineProps<{
  list: { label: string; value: string }[];
  modelValue: string;
}>();

// 雙向綁定 value
const emit = defineEmits(['update:modelValue', 'onChange']);
const inputValue = ref('');
const isOpen = ref(false);
const filteredList = computed(() => {
  if (!inputValue.value) return props.list;
  return props.list.filter(item => item.label.toLowerCase().includes(inputValue.value.toLowerCase()));
});

const selectItem = (item: { label: string; value: string }) => {
  emit('update:modelValue', item.value);
  emit('onChange', item);
  isOpen.value = false;
  inputValue.value = item.label;
};

// 當外部 value 改變時，inputValue 也要同步
watch(() => props.modelValue, (val) => {
  const found = props.list.find(item => item.value === val);
  inputValue.value = found ? found.label : '';
}, { immediate: true });

const onInput = (e: Event) => {
  inputValue.value = (e.target as HTMLInputElement).value;
  isOpen.value = true;
};

const onFocus = () => {
  isOpen.value = true;
};

const onBlur = () => {
  setTimeout(() => { isOpen.value = false; }, 150);
};

const activeIndex = ref(-1);

const onKeydown = (e: KeyboardEvent) => {
  if (!isOpen.value || !filteredList.value.length) return;
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (activeIndex.value < filteredList.value.length - 1) {
      activeIndex.value++;
    } else {
      activeIndex.value = 0;
    }
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (activeIndex.value > 0) {
      activeIndex.value--;
    } else {
      activeIndex.value = filteredList.value.length - 1;
    }
  } else if (e.key === 'Enter') {
    if (activeIndex.value >= 0 && activeIndex.value < filteredList.value.length) {
      selectItem(filteredList.value[activeIndex.value]);
    }
  }
};

watch([filteredList, isOpen], () => {
  if (!isOpen.value) {
    activeIndex.value = -1;
  } else if (filteredList.value.length > 0) {
    activeIndex.value = 0;
  }
});
</script>

<template>
  <div class="dropbox-container" style="position: relative; width: 220px;">
    <input
      class="dropbox-input"
      type="text"
      :value="inputValue"
      @input="onInput"
      @focus="onFocus"
      @blur="onBlur"
      @keydown="onKeydown"
      style="width: 100%; padding: 8px; border: 1px solid #333; border-radius: 4px; background: #222; color: #eee;"
      placeholder="Please enter keywords..."
    />
    <ul v-if="isOpen && filteredList.length" class="dropbox-list" style="position: absolute; z-index: 10; width: 100%; background: #222; border: 1px solid #333; border-radius: 4px; max-height: 180px; overflow-y: auto; margin: 0; padding: 0; list-style: none;">
      <li
        v-for="(item, idx) in filteredList"
        :key="item.value"
        class="dropbox-item"
        :class="{ 'active': idx === activeIndex }"
        @mousedown.prevent="selectItem(item)"
        style="padding: 8px; cursor: pointer; color: #eee;"
      >
        {{ item.label }}
      </li>
    </ul>
  </div>
</template>

<style scoped>
.dropbox-container { font-family: inherit; }
.dropbox-input:focus { outline: none; border-color: #888; background: #222; color: #eee; }
.dropbox-input {
  background: #222;
  color: #eee;
  border-color: #333;
}
.dropbox-list {
  background: #222;
  border-color: #333;
}
.dropbox-item:hover {
  background: #333;
  color: #fff;
}
.dropbox-item {
  color: #eee;
}
.dropbox-item.active {
  background: #444;
  color: #fff;
}
</style>