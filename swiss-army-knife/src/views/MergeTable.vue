<script setup lang="ts">
import type { IMergeTableReq } from '../helpers/dataTypes';

const emit = defineEmits<{
  (e: 'confirm', value: IMergeTableReq): void;
}>();

const handleClickMergeTable = () => {
  emit('confirm', {
    name: '',
    table1: {
      name: '',
      columns: [],
      joinOnColumns: [],
    },
    table2: {
      name: '',
      columns: [],
      joinOnColumns: [],
    },
  });
};
</script>

<template>
  <el-container>
    <el-main>Table1</el-main>
    <el-aside width="50%">Table2</el-aside>
    <el-button type="primary" @click="handleClickMergeTable" size="small">Confirm</el-button>
  </el-container>
</template>
