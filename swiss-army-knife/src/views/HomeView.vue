<script setup lang="ts">
import { reactive } from 'vue';
import CodeEditor from './CodeEditor.vue'

const data = reactive({
  code: ""
});

const onClickExecute = () => {
  console.log("code=", data.code);
}
</script>

<template>
  <el-tabs type="border-card">
    <el-tab-pane label="User">
      <codeEditor v-model="data.code" />
      <button @click="onClickExecute">Execute</button>
    </el-tab-pane>
    <el-tab-pane label="Config">Config</el-tab-pane>
    <el-tab-pane label="Role">Role</el-tab-pane>
    <el-tab-pane label="Task">Task</el-tab-pane>
  </el-tabs>
</template>
