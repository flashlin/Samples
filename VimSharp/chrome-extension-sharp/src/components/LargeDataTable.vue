<template>
  <span v-if="dt">{{ dt.tableName }}</span>
  <div style="height: 300px;">
    <!-- 顯示 keys header -->
    <div class="user header-row" v-if="dt && dt.columns && dt.columns.length > 0">
      <span v-for="col in dt.columns" :key="col.name" style="margin-right: 8px; font-weight: bold;">
        {{ col.name }}
      </span>
    </div>
    <RecycleScroller v-if="dt && dt.data" class="scroller" :items="dt.data" :item-size="32" key-field="id" v-slot="{ item }">
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