<script setup lang="ts">
import type { IHomeViewModel } from '@/types/HomeViewModel';
import { onMounted, reactive } from 'vue';
import localQueryClient from '@/apis/LocalQueryClient';
import CodeEditor from '@/components/CodeEditor.vue';

const data = reactive<IHomeViewModel>({
  searchText: "",
  tableNames: [
    "customer",
    "product"
  ]
});

function onSelectTableName(tableName: string): void {
  console.log("", tableName);
}

function search() {

}

onMounted(async () => {
  const resp = await localQueryClient.getAllTableNamesAsync();
  data.tableNames = resp.tableNames;
})
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
      </q-list>
    </q-drawer>

    <q-drawer show-if-above side="right" bordered>
      <!-- drawer right content -->
    </q-drawer>

    <q-page-container>
      <!-- drawer main content -->
      code editor
      <code-editor />
    </q-page-container>

    <q-footer elevated class="bg-grey-8 text-white">
      <q-toolbar>
        <q-toolbar-title>
          <q-avatar>
            <img src="https://cdn.quasar.dev/logo-v2/svg/logo-mono-white.svg">
          </q-avatar>
          <div>footer</div>
        </q-toolbar-title>
      </q-toolbar>
    </q-footer>

  </q-layout>
</template>
