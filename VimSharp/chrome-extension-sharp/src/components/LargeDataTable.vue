<template>
  <div style="height: 300px;">
    <!-- 顯示 keys header -->
    <div class="user header-row" v-if="list.length > 0">
      <span v-for="key in getKeys(list[0])" :key="key" style="margin-right: 8px; font-weight: bold;">
        {{ key }}
      </span>
    </div>
    <RecycleScroller class="scroller" :items="list" :item-size="32" key-field="id" v-slot="{ item }">
      <div class="user">
        <span v-for="(value, key) in item" :key="key" style="margin-right: 8px;">
          {{ value }}
        </span>
      </div>
    </RecycleScroller>
  </div>

</template>

<script lang="ts" setup>
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import { RecycleScroller } from 'vue-virtual-scroller'

// Define props using TypeScript
defineProps<{
  list: Array<{
    id: string | number
    name: string
    // 其他欄位
  }>
}>()

// 取得 keys header
function getKeys(item: Record<string, any>) {
  // 過濾掉 __ob__ 等內部屬性
  return Object.keys(item).filter(k => !k.startsWith('__'))
}
</script>

<style scoped>
.scroller {
  height: 100%;
}

.user {
  height: 32%;
  padding: 0 12px;
  display: flex;
  align-items: center;
}

.header-row {
  background: #f5f5f5;
  font-size: 15px;
  border-bottom: 1px solid #ddd;
  margin-bottom: 2px;
}
</style>