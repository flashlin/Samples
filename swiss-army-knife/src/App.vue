<script setup lang="ts">
import { ref } from 'vue';
import { storeToRefs } from 'pinia'
import { RouterView } from 'vue-router';
import { useFlashKnifeStore } from './stores/flashKnife';
const flashKnifeStore = useFlashKnifeStore();
const { fullscreenLoading } = storeToRefs(flashKnifeStore);
const { fetchAllDataTableInWebPage, showLoadingFullscreen } = flashKnifeStore;
import { ElMessageBox } from 'element-plus'

// const data = reactive({
//   jsonContent: jsonKnifeStore.jsonContent,
// })

const dialogVisible = ref(false)
const handleClose = (done: () => void) => {
  ElMessageBox.confirm('Are you sure to close this dialog?')
    .then(() => {
      done()
    })
    .catch(() => {
      // catch error
    })
}

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

const handleClickFlashIcon = () => {
  if (dialogVisible.value == true) {
    dialogVisible.value = false;
  } else {
    dialogVisible.value = true;
  }
}
</script>

<template>
  <div class="flash-sidebar flash-icon" @click="handleClickFlashIcon">F</div>
  <div v-loading.fullscreen.lock="fullscreenLoading"></div>

  <el-dialog v-model="dialogVisible" title="Tips" width="30%" :before-close="handleClose">
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

    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="dialogVisible = false">
          Confirm
        </el-button>
      </span>
    </template>
  </el-dialog>
</template>

<style scoped>
.flash-icon {
  display: inline-block;
  width: 32px;
  height: 32px;
  font-size: 24px;
  font-weight: bold;
  color: gold;
  text-align: center;
  line-height: 32px;
  background: none;
  cursor: pointer;
}

.flash-sidebar {
  margin: 0;
  padding: 0;
  width: 30px;
  height: 30px;
  background-color: #04AA6D;
  overflow: auto;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 999;
  background: none;
  overflow: hidden;
}

.flash-sidebar expand {
  width: 410px;
  height: 510px;
}

.flash-sidebar a {
  display: block;
  color: black;
  padding: 16px;
  text-decoration: none;
}

.flash-sidebar a.active {
  background-color: #04AA6D;
  color: white;
}

.flash-sidebar a:hover:not(.active) {
  background-color: #555;
  color: white;
}
</style>