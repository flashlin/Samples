<script setup lang="ts">
import type { IHomeViewModel } from '@/types/HomeViewModel';
import { onMounted, reactive } from 'vue';
import localQueryClient from '@/apis/LocalQueryClient';

const data = reactive<IHomeViewModel>({
  searchText: "",
  tableNames: [
    "customer",
    "product"
  ]
});

function onSelectTableName(tableName: string): void {
  console.log("", tableName);
}

function search() {
  
}

onMounted(async () => {
  const resp = await localQueryClient.getAllTableNamesAsync();
  data.tableNames = resp.tableNames;
})
</script>

<template>
  <h6>Menu</h6>
  <q-input
      outlined
      placeholder="Search"
      v-model="data.searchText"
      @input="search"
    />
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
