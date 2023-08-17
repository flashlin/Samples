<script setup lang="ts">
import { ref } from 'vue';
import { storeToRefs } from 'pinia'
import { RouterView } from 'vue-router';
import { useFlashKnifeStore } from './stores/flashKnife';
const flashKnifeStore = useFlashKnifeStore();
const { fullscreenLoading } = storeToRefs(flashKnifeStore);
const { fetchAllDataTableInWebPage, showLoadingFullscreen } = flashKnifeStore;

// const data = reactive({
//   jsonContent: jsonKnifeStore.jsonContent,
// })

const activeIndex = ref('1');
const handleSelect = (key: string, keyPath: string[]) => {
  if (key == "FetchDataTableInWebPage") {
    console.log(keyPath)
    showLoadingFullscreen(true);
    fetchAllDataTableInWebPage();
    setTimeout(() => {
      showLoadingFullscreen(false);
    }, 1000);
  }
};
</script>

<template>
  <div v-loading.fullscreen.lock="fullscreenLoading"></div>
  <el-menu :default-active="activeIndex" class="el-menu-demo" mode="horizontal" background-color="#545c64"
    text-color="#fff" active-text-color="#ffd04b" @select="handleSelect">
    <el-menu-item index="1">Processing Center</el-menu-item>
    <el-sub-menu index="2">
      <template #title>Workspace</template>
      <el-menu-item index="2-1">item one</el-menu-item>
      <el-menu-item index="2-2">item two</el-menu-item>
      <el-menu-item index="2-3">item three</el-menu-item>
      <el-sub-menu index="2-4">
        <template #title>item four</template>
        <el-menu-item index="2-4-1">item one</el-menu-item>
        <el-menu-item index="2-4-2">item two</el-menu-item>
        <el-menu-item index="2-4-3">item three</el-menu-item>
      </el-sub-menu>
    </el-sub-menu>
    <el-menu-item index="3" disabled>Info</el-menu-item>
    <el-menu-item index="FetchDataTableInWebPage">FetchDataTableInWebPage</el-menu-item>
  </el-menu>
  <RouterView />
</template>

<style scoped></style>