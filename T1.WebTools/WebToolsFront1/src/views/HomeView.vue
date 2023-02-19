<script setup lang="ts">
import type { IHomeViewModel } from '@/types/HomeViewModel';
import { onMounted, reactive } from 'vue';
import localQueryClient from '@/apis/LocalQueryClient';

const data = reactive<IHomeViewModel>({
  tableNames: [
    "customer",
    "product"
  ]
});

function onSelectTableName(tableName: string): void {
  console.log("", tableName);
}

onMounted(async () => {
  const resp = await localQueryClient.getAllTableNamesAsync();
  data.tableNames = resp.tableNames;
})
</script>

<template>
  <h6>Menu</h6>
  <q-list bordered separator>
    <template v-for="tableName in data.tableNames">
      <q-item clickable v-ripple @click="() => onSelectTableName(tableName)">
        <q-item-section>
          {{ tableName }}
        </q-item-section>
      </q-item>
    </template>
  </q-list>
</template>
