<template>
  <span v-if="dt">{{ dt.tableName }}</span>
  <div class="w-full max-w-4xl mx-auto mt-8" style="height: 300px;">
    <!-- Header row as grid (dynamic columns) -->
    <div
      v-if="dt && dt.columns && dt.columns.length > 0"
      class="grid px-4 py-2 bg-gray-800 text-white font-semibold text-sm rounded-t-md dark:bg-gray-900 dark:text-gray-100"
      :class="`grid-cols-${dt.columns.length + 1} gap-4`"
    >
      <div v-for="col in dt.columns" :key="col.name">{{ col.name }}</div>
      <div>操作</div>
    </div>

    <!-- Data rows with virtual scroll (dynamic columns) -->
    <RecycleScroller
      v-if="dt && dt.data"
      class="divide-y divide-gray-700 dark:divide-gray-800 scroller"
      :items="dt.data"
      :item-size="48"
      key-field="id"
      v-slot="{ item }"
    >
      <div
        class="grid px-4 py-3 bg-gray-900 text-gray-100 hover:bg-gray-800 transition dark:bg-gray-950 dark:hover:bg-gray-800"
        :class="`grid-cols-${dt.columns.length + 1} gap-4`"
      >
        <div v-for="col in dt.columns" :key="col.name">{{ item[col.name] }}</div>
        <div>
          <button class="px-2 py-1 text-sm text-white bg-blue-600 rounded hover:bg-blue-500">編輯</button>
        </div>
      </div>
    </RecycleScroller>
  </div>
</template>

<script lang="ts" setup>
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import { RecycleScroller } from 'vue-virtual-scroller'
import { DataTable as DataTableType } from '@/tools/dataTypes'

// Define props using TypeScript
defineProps<{
  dt: DataTableType | null
}>()
</script>

<style scoped>
.scroller {
  height: 100%;
}
</style>