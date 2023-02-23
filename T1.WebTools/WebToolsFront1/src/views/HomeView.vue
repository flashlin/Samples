<script setup lang="ts">
import type { IHomeViewModel } from '@/types/HomeViewModel';
import { onActivated, onMounted, reactive, ref } from 'vue';
import CodeEditor from '@/components/CodeEditor.vue';
import TerminalUi from '@/components/TerminalUi.vue';
import { useAppState } from '@/stores/appState';
import type { QUploaderFactoryFn, QUploaderFactoryObject } from 'quasar';
import Hotkeys from 'vue-hotkeys-rt';
import type { ITerminalUiProxy } from '@/components/TerminalUiModel';

const appState = useAppState();

const data = reactive<IHomeViewModel>({
  searchText: "",
  tableNames: [],
  code: `CREATE TABLE (\n[id] int IDENTITY(1,1),\n[name] nvarchar(50))\n`
});

const terminalRef = ref<ITerminalUiProxy>();

function onSelectTableName(tableName: string): void {
  console.log("", tableName);
}

function search() {

}


function uploadFileFactory(files: readonly File[]): Promise<QUploaderFactoryObject> {
  return new Promise((resolve, reject) => {
    const token = "myToken";
    resolve({
      url: `http://127.0.0.1:${appState.appPort}/api/LocalQueryApi/uploadFiles`,
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

async function onTriggeredEventHandler(payload: IHotkey) {
  const term = terminalRef.value!;
  if (payload.keyString == 'R') {
    //term.writeln(payload.keyString);
    //term.writeln(data.code);
    const localQueryClient = appState.localQueryClient!;
    const result = await localQueryClient.queryRawSql({
      sql: data.code
    });
    term.writeln(result.errorMessage);
    term.writeln(JSON.stringify(result.data));
    updateTableNames();
  }
}

async function updateTableNames(){
  const localQueryClient = appState.localQueryClient!;
  const resp = await localQueryClient.getAllTableNamesAsync();
  data.tableNames.splice(0, data.tableNames.length, ...resp.tableNames);
}

updateTableNames();
</script>

<template>
  <Hotkeys :shortcuts="['R', 'A', 'ArrowLeft', 'ArrowRight']" @triggered="onTriggeredEventHandler" />
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
      <code-editor v-model:code="data.code" />
      <terminal-ui ref="terminalRef"></terminal-ui>
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
