<script setup lang="ts">
import type { IHomeViewModel } from '@/types/HomeViewModel';
import { reactive } from 'vue';

const data = reactive<IHomeViewModel>({
  tablesName: [
    "customer",
    "product"
  ]
});

function onSelectTableName(tableName: string): void {
  console.log("", tableName);
}
</script>

<template>
  <h6>Menu</h6>
  <q-list bordered separator>
    <template v-for="tableName in data.tablesName">
      <q-item clickable v-ripple @click="() => onSelectTableName(tableName)">
        <q-item-section>
          {{ tableName }}
        </q-item-section>
      </q-item>
    </template>
  </q-list>
</template>
