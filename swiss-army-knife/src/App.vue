<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue';
import { storeToRefs } from 'pinia';
//import { RouterView } from 'vue-router';
import { useFlashKnifeStore, type IPrepareImportDataTable } from './stores/flashKnife';
import CodeEditor from './views/CodeEditor.vue';
import DataTable from './components/DataTable.vue';
import { QuerySqliteService, SqliteDb } from './helpers/sqliteDb';
import { MessageTypes, type IDataTable, type MessageType } from './helpers/dataTypes';
import { ElNotification } from 'element-plus';
import { UploadFilled } from '@element-plus/icons-vue'

import { exportToCsv, getCurrentTime, parseCsvContentToObjectArray, readFileContentAsync } from './helpers/dataHelper';

const flashKnifeStore = useFlashKnifeStore();
const { fullscreenLoading, tableListInWebPage } = storeToRefs(flashKnifeStore);
const { fetchAllDataTableInWebPage, showLoadingFullscreen } = flashKnifeStore;

interface IData {
  tableName: string;
  code: string;
  tableNames: string[];
  dataTable: IDataTable;
  tableStructure: IDataTable;
}
const db = new SqliteDb();
const queryService = new QuerySqliteService(db);
const data = reactive<IData>({
  tableName: 'tb0',
  code: 'select Company,Contact,Country from tb0',
  tableNames: [],
  dataTable: {
    columnNames: [],
    rows: [],
  },
  tableStructure: {
    columnNames: [],
    rows: [],
  }
});
const dialogVisible = ref(false);

const tableList = computed(() => {
  let idx = -1;
  const tableData = tableListInWebPage.value.map((x: IPrepareImportDataTable) => {
    idx++;
    return {
      idx: idx,
      tableName: x.tableName,
      columns: x.dataTable.columnNames.join(','),
    };
  });
  return tableData;
});

const tableNamesTable = computed(() => {
  const result: object[] = [];
  data.tableNames.forEach(tableName => {
    result.push({
      'tableName': tableName
    });
  });
  return result;
});

const notify = (messageType: MessageType, message: string) => {
  ElNotification({
    title: 'Success',
    message: message,
    type: messageType,
    position: 'top-right',
  });
};

const onClickExportToCsv = () => {
  const time = getCurrentTime();
  exportToCsv(`flash-data-${time}`, data.dataTable.rows);
}

const onClickExecute = async () => {
  try {
    const result = db.queryDataTableByBindParams(data.code);
    notify(MessageTypes.Success, `Executed`);
    data.dataTable = result;
  } catch (e1) {
    if (e1 instanceof Error) {
      notify(MessageTypes.Error, e1.message);
    } else {
      notify(MessageTypes.Error, `${e1}`);
    }
  }
};

const handleClickTableStructure = (tableName: string) => {
  data.tableStructure = queryService.getTableFieldsInfo(tableName);
  notify(MessageTypes.Success, `get ${tableName} structure`)
};

const handleClickImportWebPageTable = async (idx: number) => {
  const table = tableList.value[idx];
  const rawTable = tableListInWebPage.value[idx];
  const count = db.importTable(table.tableName, rawTable.dataTable.rows);
  notify(MessageTypes.Success, `import ${table.tableName}, Data Count=${count}`);
  queryAllTableNames();
};

const handleDialogClose = (done: () => void) => {
  closeFlashIcon();
  done();
};

const queryAllTableNames = () => {
  data.tableNames = queryService.getAllTableNames();
};

const activeIndex = ref('2');
const handleSelect = (key: string, keyPath: string[]) => {
  if (key == 'FetchTableInWebPage') {
    showLoadingFullscreen(true);
    fetchAllDataTableInWebPage();
    showLoadingFullscreen(false);
    notify(MessageTypes.Success, `${keyPath}`);
    return;
  }

  if (key == 'ExportToCsv') {
    onClickExportToCsv();
    return;
  }

  if (key == 'ImportQueryData') {
    onClickImportQueryData();
    return;
  }

  if (key == 'QueryAllTableNames') {
    queryAllTableNames();
    return;
  }
};

const toggleTheme = (toggle: boolean) => {
  const htmlElement = document.querySelector('html')!;
  if (toggle) {
    htmlElement.classList.add('dark');
  } else {
    htmlElement.classList.remove('dark');
  }
}

const closeFlashIcon = () => {
  dialogVisible.value = false;
  toggleTheme(false);
}

const handleClickFlashIcon = () => {
  if (dialogVisible.value == true) {
    closeFlashIcon();
  } else {
    dialogVisible.value = true;
    toggleTheme(true);
  }
};

const onClickImportQueryData = () => {
  db.importTable(data.tableName, data.dataTable.rows);
}

const onHandleBeforeUpload = async (file: File) => {
  const fileContent = await readFileContentAsync(file);
  const uploadDataTable = parseCsvContentToObjectArray(fileContent);
  data.dataTable = uploadDataTable;
  notify(MessageTypes.Success, `Upload csv File ${file.name}`);
  return false;
}

const handleOnDeleteTable = (tableName: string) => {
  db.dropTable(tableName);
  notify(MessageTypes.Success, `Drop Table ${tableName}`);
  queryAllTableNames();
}

const handleKeyPress = (event: KeyboardEvent) => {
  if (dialogVisible.value == false) {
    return;
  }
  if (event.key === 'F8') {
    onClickExecute();
  }
};

onMounted(async () => {
  await db.openAsync();
  window.addEventListener('keydown', handleKeyPress);
});

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyPress);
});
</script>

<template>
  <div>
    <div class="flash-sidebar flash-icon" @click="handleClickFlashIcon">F</div>
    <div v-loading.fullscreen.lock="fullscreenLoading"></div>
    <el-dialog v-model="dialogVisible" title="FlashKnife V0.1" top="32px" width="98%" :before-close="handleDialogClose">

      <!-- TOP MENU -->
      <el-menu :default-active="activeIndex" class="el-menu-demo" mode="horizontal" background-color="#545c64"
        text-color="#fff" active-text-color="#ffd04b" @select="handleSelect">
        <el-sub-menu index="">
          <template #title>Database</template>
          <el-menu-item index="QueryAllTableNames">Query All Table Names</el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="2">
          <template #title>Import Data</template>
          <el-menu-item index="FetchTableInWebPage">Fetch Table In WebPage</el-menu-item>
          <el-menu-item index="ImportQueryData">Import QueryData</el-menu-item>
          <el-sub-menu index="2-4">
            <template #title>item four</template>
            <el-menu-item index="2-4-1">item one</el-menu-item>
            <el-menu-item index="2-4-2">item two</el-menu-item>
            <el-menu-item index="2-4-3">item three</el-menu-item>
          </el-sub-menu>
        </el-sub-menu>
        <el-menu-item index="ExportToCsv">ExportToCsv</el-menu-item>
      </el-menu>

      <el-tabs type="border-card">
        <el-tab-pane label="Sqlite">
          <div class="common-layout">
            <el-container>
              <el-main>
                <CodeEditor v-model="data.code" />
              </el-main>
              <el-aside width="200px">
                <el-table :data="tableNamesTable" stripe style="width: 100%">
                  <el-table-column prop="tableName" label="tableName" width="180" />
                  <el-table-column fixed="right" label="Operations" width="120">
                    <template #default="scope">
                      <el-button link type="primary" @click="handleOnDeleteTable(scope.row.tableName)"
                        size="small">Delete</el-button>
                      <el-button link type="primary" @click="handleClickTableStructure(scope.row.tableName)"
                        size="small">Structure</el-button>
                    </template>
                  </el-table-column>
                </el-table>

              </el-aside>
            </el-container>
          </div>

          <el-form label-width="120px">
            <el-form-item label="Import Table Name">
              <el-input v-model="data.tableName" placeholder="Please input import table name" />
            </el-form-item>
            <DataTable v-model="data.dataTable" />
          </el-form>

          <el-divider />
          <DataTable v-model="data.tableStructure" />
          <el-divider />

          <!-- Query Table In WebPage -->
          <el-table :data="tableList" stripe style="width: 100%">
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
                <el-button link type="primary" size="small"
                  @click="handleClickImportWebPageTable(scope.row.idx)">Import</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
        <el-tab-pane label="Upload Area">
          <el-upload class="upload-demo" drag action="" multiple accept=".csv" :before-upload="onHandleBeforeUpload">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              Drop csv file here or <em>click to upload</em><br>
              When uploaded, you can view the data in the query result area.
            </div>
            <template #tip>
              <div class="el-upload__tip">
                csv files with a size less than 500kb
              </div>
            </template>
          </el-upload>
          SELECT name FROM sqlite_master WHERE type='table'
          SELECT customer.id, customer.name, product.pname, product.price
          FROM customer LEFT JOIN product ON customer.id = product.id
        </el-tab-pane>
      </el-tabs>

      <template #footer>
        <span class="dialog-footer">
          <el-button type="primary" @click="closeFlashIcon"> Close </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
div.dark {
  --el-bg-color: #3d3d3d;
}

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
