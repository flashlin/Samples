<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue';
import { storeToRefs } from 'pinia';
//import { RouterView } from 'vue-router';
import { useFlashKnifeStore, type IPrepareImportDataTable } from './stores/flashKnife';
import CodeEditor from './views/CodeEditor.vue';
import DataTable from './components/DataTable.vue';
import { SqliteDb } from './helpers/sqliteDb';

const flashKnifeStore = useFlashKnifeStore();
const { fullscreenLoading, dataTableListInWebPage } = storeToRefs(flashKnifeStore);
const { fetchAllDataTableInWebPage, showLoadingFullscreen } = flashKnifeStore;

interface IData {
  code: string;
  dataTable: {
    headerNames: string[];
    rows: any[];
  }
}

const data = reactive<IData>({
  code: 'select Company,Contact,Country from tb0',
  dataTable: {
    headerNames: [],
    rows: [],
  }
});
const dialogVisible = ref(false);

const onClickExecute = async () => {
  console.log('code=', data.code);
  const db = new SqliteDb();
  await db.openAsync();
  const result = db.query(data.code);
  const rows: any[] = [];
  const headerNames: string[] = [];
  result.forEach(row => {
    const obj: any = {}
    for (let idx = 0; idx < row.length; idx++) {
      headerNames[idx] = `field${idx}`
      obj[`field${idx}`] = row[idx];
    }
    rows.push(obj);
  });
  data.dataTable.headerNames = headerNames;
  data.dataTable.rows = rows;
  db.close();
};


const dataTableList = computed(() => {
  let idx = -1;
  const tableData = dataTableListInWebPage.value.map((x: IPrepareImportDataTable) => {
    idx++;
    return {
      idx: idx,
      tableName: x.tableName,
      columns: x.dataTable.headerNames.join(','),
    };
  });
  return tableData;
});

const handleClickImport = async (idx: number) => {
  const table = dataTableList.value[idx];
  const rawTable = dataTableListInWebPage.value[idx];
  const db = new SqliteDb();
  await db.openAsync();
  db.importTable(table.tableName, rawTable.dataTable.rows);
  const t1 = db.query(data.code);
  console.log('q', t1)
  db.close();
};

const handleClose = (done: () => void) => {
  done();
};

const activeIndex = ref('1');
const handleSelect = (key: string, keyPath: string[]) => {
  if (key == 'FetchDataTableInWebPage') {
    console.log(keyPath);
    showLoadingFullscreen(true);
    fetchAllDataTableInWebPage();
    showLoadingFullscreen(false);
  }
};

const handleClickFlashIcon = () => {
  if (dialogVisible.value == true) {
    dialogVisible.value = false;
  } else {
    dialogVisible.value = true;
  }
};

const handleKeyPress = (event: KeyboardEvent) => {
  if (dialogVisible.value == false) {
    return;
  }
  if (event.key === 'F8') {
    onClickExecute();
  }
};

onMounted(() => {
  window.addEventListener('keydown', handleKeyPress);
});

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyPress);
});
</script>

<template>
  <div class="flash-sidebar flash-icon" @click="handleClickFlashIcon">F</div>
  <div v-loading.fullscreen.lock="fullscreenLoading"></div>
  <el-dialog v-model="dialogVisible" title="FlashKnife" top="32px" width="98%" :before-close="handleClose">
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
    RouterView

    <el-tabs type="border-card">
      <el-tab-pane label="Sqlite">
        <CodeEditor v-model="data.code" />
        <DataTable v-model="data.dataTable" />
        <el-table :data="dataTableList" stripe style="width: 100%">
          <el-table-column label="tableName" width="180">
            <template #default="scope">
              <div style="display: flex; align-items: center">
                <el-input v-model="scope.row.tableName" placeholder="Please input" />
              </div>
            </template>
          </el-table-column>
          <el-table-column prop="columns" label="columns" width="800" />
          <el-table-column fixed="right" label="Operations" width="120">
            <template #default="scope">
              <el-button link type="primary" size="small" @click="handleClickImport(scope.row.idx)">Import</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
      <el-tab-pane label="Config">Config</el-tab-pane>
      <el-tab-pane label="Role">Role</el-tab-pane>
      <el-tab-pane label="Task">Task</el-tab-pane>
    </el-tabs>

    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="dialogVisible = false"> Confirm </el-button>
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
  background-color: #04aa6d;
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
  background-color: #04aa6d;
  color: white;
}

.flash-sidebar a:hover:not(.active) {
  background-color: #555;
  color: white;
}
</style>
