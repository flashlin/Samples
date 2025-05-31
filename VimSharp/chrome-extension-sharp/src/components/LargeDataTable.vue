<template>
  <div class="w-full max-w-4xl mx-auto mt-8" style="height: 300px;">
    <!-- Header row as grid (dynamic columns) -->
    <span v-if="dt">{{ dt.tableName }}</span>
    <div
      v-if="dt && dt.columns && dt.columns.length > 0"
      class="grid px-4 py-2 bg-gray-800 text-white font-semibold text-sm rounded-t-md dark:bg-gray-900 dark:text-gray-100"
      :class="`grid-cols-${dt.columns.length} gap-4`"
    >
      <div v-for="col in dt.columns" :key="col.name">{{ col.name }}</div>
    </div>

    <!-- Data rows with virtual scroll (dynamic columns) -->
    <RecycleScroller
      v-if="dt && dt.data"
      class="divide-y divide-gray-700 dark:divide-gray-800 scroller"
      :items="dt.data"
      :item-size="48"
      :key-field="keyField"
      v-slot="{ item }"
    >
      <div
        class="grid px-4 py-3 bg-gray-900 text-gray-100 hover:bg-gray-800 transition dark:bg-gray-950 dark:hover:bg-gray-800"
        :class="`grid-cols-${dt.columns.length} gap-4`"
      >
        <div v-for="col in dt.columns" :key="col.name">{{ item[col.name] }}</div>
      </div>
    </RecycleScroller>
  </div>
</template>

<script lang="ts" setup>
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import { RecycleScroller } from 'vue-virtual-scroller'
import { DataTable as DataTableType } from '@/tools/dataTypes'
import { computed, watchEffect } from 'vue'

// Define props using TypeScript
const props = defineProps<{
  dt: DataTableType | null,
  keyField?: string | null
}>()
const dt = computed(() => props.dt)

// 檢查 columns 是否有 id 欄位（改為判斷 props.keyField 是否為 null）
const hasIdColumn = computed(() => props.keyField !== null && props.keyField !== undefined)

// 若沒有 id 欄位，為每筆 data 加上 _id（流水號），並在 columns 加上 _id 欄位
watchEffect(() => {
  if (!dt.value) return
  if (!dt.value.columns) return
  if (!hasIdColumn.value) {
    // 若 columns 沒有 _id 欄位，則加上
    if (!dt.value.columns.some((col: any) => col.name === '_id')) {
      dt.value.columns.push({ name: '_id', type: 'INTEGER' })
    }
    // 為每筆 data 加上 _id
    if (dt.value.data) {
      dt.value.data.forEach((row: any, idx: number) => {
        if (row._id === undefined) row._id = idx + 1
      })
    }
  }
})

// keyField 決定邏輯：props.keyField 為 null 或 undefined 則用 _id，否則用 props.keyField
const keyField = computed(() => (props.keyField == null ? '_id' : props.keyField))
</script>

<style scoped>
.scroller {
  height: 100%;
}
</style>