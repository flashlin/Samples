<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router';
import { QBtn } from 'quasar';

import { getCurrentInstance } from "@vue/runtime-core";
// import { onMounted } from 'vue';
// const {
//     appContext: {
//         config: {
//             globalProperties: { app },
//         },
//     },
// } = getCurrentInstance()!;

</script>

<template>
  <q-layout view="lHh Lpr fFf">
    <q-header elevated class="bg-primary text-white">
      <q-toolbar>
        <q-btn dense flat round icon="menu" aria-label="Menu" />
        <q-toolbar-title>
          LocalQuery App v1.0
        </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-page-container>
      <RouterView />
    </q-page-container>
  </q-layout>
</template>

<style lang="sass" scoped>
</style>
