<script lang="ts" setup>
import { DataTable } from '@/tools/dataTypes'
import { computed } from 'vue'

const props = defineProps<{ value: DataTable }>()

// 只取前 100 筆資料
const rows = computed(() => (props.value?.data || []).slice(0, 100))
const columns = computed(() => props.value?.columns || [])
</script>

<template>
  <div class="relative overflow-x-auto">
    <table class="w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
      <thead class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
        <tr>
          <th v-for="col in columns" :key="col.name" class="px-6 py-3">
            {{ col.name }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, rowIdx) in rows" :key="rowIdx" class="bg-white border-b dark:bg-gray-800 dark:border-gray-700 border-gray-200">
          <td v-for="col in columns" :key="col.name" class="px-6 py-4">
            {{ row[col.name] }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>