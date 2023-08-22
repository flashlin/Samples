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
    <el-header>Header</el-header>
    <el-aside width="50%">
      <select name="cars" size="5">
        <option value="volvo">Volvo</option>
        <option value="saab">Saab</option>
        <option value="mercedes">Mercedes</option>
        <option value="audi">Audi</option>
      </select>
    </el-aside>
    <el-aside width="50%">
      <select name="cars" size="5">
        <option value="volvo">Volvo</option>
        <option value="saab">Saab</option>
        <option value="mercedes">Mercedes</option>
        <option value="audi">Audi</option>
      </select>
    </el-aside>
    <el-main>
      <select name="cars" size="5">
        <option value="volvo">Volvo</option>
        <option value="saab">Saab</option>
        <option value="mercedes">Mercedes</option>
        <option value="audi">Audi</option>
      </select>
    </el-main>
    <el-footer>
      <el-button type="primary" @click="handleClickMergeTable" size="small">Confirm</el-button>
    </el-footer>
  </el-container>
</template>
