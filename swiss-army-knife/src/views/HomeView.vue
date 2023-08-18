<script setup lang="ts">
import { reactive } from 'vue';
import { useFlashKnifeStore, type IPrepareImportDataTable } from '../stores/flashKnife';
//import CodeEditor from './CodeEditor.vue'
import { storeToRefs } from 'pinia';

const flashKnifeStore = useFlashKnifeStore();
const { dataTableListInWebPage } = storeToRefs(flashKnifeStore);

const data = reactive({
  code: ""
});

const onClickExecute = () => {
  console.log("code=", data.code);
}

const tableData = dataTableListInWebPage.value.map((x: IPrepareImportDataTable) => {
  return {
    tableName: x.tableName,
    columns: x.dataTable.headerNames.join(",")
  }
})
</script>

<template>
  <el-tabs type="border-card">
    <el-tab-pane label="User">
      <codeEditor v-model="data.code" />
      <button @click="onClickExecute">Execute</button>
      <el-table :data="tableData" stripe style="width: 100%">
        <el-table-column prop="tableName" label="tableName" width="180" />
        <el-table-column prop="columns" label="columns" width="800" />
      </el-table>
    </el-tab-pane>
    <el-tab-pane label="Config">Config</el-tab-pane>
    <el-tab-pane label="Role">Role</el-tab-pane>
    <el-tab-pane label="Task">Task</el-tab-pane>
  </el-tabs>
</template>
