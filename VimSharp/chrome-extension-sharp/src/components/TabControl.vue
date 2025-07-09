<script setup lang="ts">
export interface TabItem {
  id: string
  name: string
}

defineProps<{
  tabList: TabItem[]
  activeTab: string
}>()

const emit = defineEmits(['update:activeTab'])

function selectTab(tabId: string) {
  emit('update:activeTab', tabId)
}
</script>

<template>
  <div class="tabs">
    <button 
      v-for="tab in tabList"
      :key="tab.id"
      class="tab-button" 
      :class="{ active: activeTab === tab.id }"
      @click="selectTab(tab.id)"
    >
      {{ tab.name }}
    </button>
  </div>
  <div class="tab-panels">
    <slot :name="activeTab" />
  </div>
</template>

<style scoped>
.tabs {
  display: flex;
  background-color: #252526;
  border-bottom: 1px solid #333;
}

.tab-button {
  padding: 10px 20px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 14px;
  color: #969696;
  transition: all 0.3s ease;
  border-right: 1px solid #333;
  position: relative;
  overflow: hidden;
}

.tab-button:hover {
  background-color: #2d2d2d;
  color: #ffffff;
}

.tab-button.active {
  background-color: #1e1e1e;
  color: #4CAF50;
  border-bottom: 2px solid #4CAF50;
}

.tab-button::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: #4CAF50;
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.tab-button.active::after {
  transform: scaleX(1);
}

.tab-panels {
  background-color: #1e1e1e;
}
</style> 