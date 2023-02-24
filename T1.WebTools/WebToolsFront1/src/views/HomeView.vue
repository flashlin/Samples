<script setup lang="ts">
import { onActivated, onMounted, reactive, ref } from 'vue';
import type { IHomeViewModel } from '@/types/HomeViewModel';
import CodeEditor from '@/components/CodeEditor.vue';
import TerminalUi from '@/components/TerminalUi.vue';
import { useAppStore } from '@/stores/appStore.js';
import type { QUploaderFactoryFn, QUploaderFactoryObject } from 'quasar';
import type { ITerminalUiProxy } from '@/components/TerminalUiModel';
import { storeToRefs } from 'pinia';

const appStore = useAppStore();
const appState = storeToRefs(appStore);
const localQueryClient = appStore.getLocalQueryClient();

const data = reactive<IHomeViewModel>({
  localFile: '',
  searchText: '',
  tableNames: [],
  code: `CREATE TABLE (\n[id] int IDENTITY(1,1),\n[name] nvarchar(50))\n`,
  tabName: 'a1',
});

const terminalRef = ref<ITerminalUiProxy>();

function onSelectTableName(tableName: string): void {
  console.log("", tableName);
}

function search() {

}


function uploadFileFactory(files: readonly File[]): Promise<QUploaderFactoryObject> {
  return new Promise((resolve, reject) => {
    console.log('upload appPort', appState.appPort.value);
    const token = "myToken";
    resolve({
      url: `http://127.0.0.1:${appState.appPort.value}/api/LocalQueryApi/uploadFiles`,
      method: 'POST',
      headers: [
        { name: 'Authorization', value: `Bearer ${token}` }
      ]
    })
  })
}

interface IHotkey {
  keyString: string;
  keyCode: number;
}

function clear() {
    const term = terminalRef.value!;
    term.clear();
}

function write(msg: string) {
    const term = terminalRef.value!;
    term.write(msg);
}


function writeln(msg: string) {
    const term = terminalRef.value!;
    term.writeln(msg);
}


async function updateTableNames() {
  const resp = await localQueryClient.getAllTableNamesAsync();
  data.tableNames.splice(0, data.tableNames.length, ...resp.tableNames);
}

async function onImportLocalFile() {
  await localQueryClient.importLocalFile({
    filePath: data.localFile
  });
  updateTableNames();
  clear();
  writeln(`import ${data.localFile} success`);
}

onMounted(() => {
  window.addEventListener('keydown', async (event: KeyboardEvent) => {
    if (event.ctrlKey && event.key === 'r') {
      event.preventDefault();
      const result = await localQueryClient.queryRawSql({
        sql: data.code
      });
      clear();
      writeln('' + new Date());
      
      const csvSheet = result.csvSheet;
      let count = 0;
      if( csvSheet.rows != null ) {
        count = csvSheet.rows.length;
      }
      writeln(`count: ${count}`);
      writeln(result.errorMessage);

      //term.writeln(JSON.stringify(result.csvSheet.headers));
      const headers = result.csvSheet.headers;
      for(let header of headers) {
        let cols = 0;
        write(header.name);
        if( cols < headers.length-1 ){
          write(',');
        }
        cols++;
      }
      writeln('');

      for(let row of result.csvSheet.rows)
      {
        let cols = 0;
        for(let header of result.csvSheet.headers) {
          write(row[header.name]);
          if( cols < result.csvSheet.headers.length-1 ){
            write(',');
          }
          cols++;
        }
        writeln('');
      }

      updateTableNames();
    }
  });
});

updateTableNames();
</script>

<template>
  <q-layout view="hHh lpR fFf">

    <!-- <q-header elevated class="bg-primary text-white" height-hint="98">
                                                                                                                                                          <q-toolbar>
                                                                                                                                                            <q-btn dense flat round icon="menu" />
                                                                                                                                                            <q-toolbar-title>
                                                                                                                                                                  <q-avatar>
                                                                                                                                                                    <img src="https://cdn.quasar.dev/logo-v2/svg/logo-mono-white.svg">
                                                                                                                                                                  </q-avatar>
                                                                                                                                                              Title
                                                                                                                                                            </q-toolbar-title>

                                                                                                                                                      <q-btn dense flat round icon="menu" />
                                                                                                                                                    </q-toolbar>

                                                                                                                                                    <q-tabs align="left">
                                                                                                                                                      <q-route-tab to="/page1" label="Page One" />
                                                                                                                                                        <q-route-tab to="/page2" label="Page Two" />
                                                                                                                                                            <q-route-tab to="/page3" label="Page Three" />
                                                                                                                                                          </q-tabs>
                                                                                                                                                              </q-header> -->

    <q-drawer show-if-above side="left" bordered>
      <!-- drawer left content -->
      <q-input outlined placeholder="Search" v-model="data.searchText" @input="search">
        <template v-slot:prepend>
          <q-icon name="search" />
        </template>
      </q-input>
      <q-list bordered separator>
        <template v-for="tableName in data.tableNames">
          <q-item clickable v-ripple @click="() => onSelectTableName(tableName)">
            <q-item-section>
              {{ tableName }}
            </q-item-section>
          </q-item>
        </template>
        <div class="q-pa-md">
          <q-uploader multiple batch label="csv/excel multiple upload" :factory="uploadFileFactory" dark
            style="max-width: 250px" />
        </div>
      </q-list>
    </q-drawer>

    <q-drawer show-if-above side="right" bordered>
      <!-- drawer right content -->
    </q-drawer>

    <q-page-container>
      <!-- drawer main content -->
      <q-input v-model="data.localFile" label="import local file" />
      <q-btn color="secondary" label="Import" @click="onImportLocalFile"/>
      <code-editor v-model:code="data.code" />
      <q-tabs align="justify" tabIndex="0">
        <q-tab>Result</q-tab>
        <q-tab>Grid</q-tab>
      </q-tabs>
      <q-tab-panels v-model="data.tabName" style="height: 250px;">
        <q-tab-panel name="a1">
          <terminal-ui ref="terminalRef" style="height: 250px;"></terminal-ui>
        </q-tab-panel>
        <q-tab-panel name="a2">
          cdcdcdcdcd
        </q-tab-panel>
      </q-tab-panels>
    </q-page-container>

    <!-- <q-footer elevated class="bg-grey-8 text-white">
                                                                                                                                                            <q-toolbar>
                                                                                                                                                              <q-toolbar-title>
                                                                                                                                                                <q-avatar>
                                                                                                                                                                  <img src="https://cdn.quasar.dev/logo-v2/svg/logo-mono-white.svg">
                                                                                                                                                                </q-avatar>
                                                                                                                                                                <div>footer</div>
                                                                                                                                                              </q-toolbar-title>
                                                                                                                                                            </q-toolbar>
                                                                                                                                                          </q-footer> -->
  </q-layout>
</template>
