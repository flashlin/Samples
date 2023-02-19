<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router';

</script>

<template>
  llll
  <q-layout view="lHh Lpr fFf">
    <q-header elevated class="bg-primary text-white">
      <q-toolbar>
        <q-btn dense flat round icon="menu" aria-label="Menu" />
        <q-toolbar-title>
          LocalQuery v1.0
        </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-drawer side="left">
      <q-list>
        <q-item to="/" clickable>
          <q-item-section>
            <q-item-label>Home</q-item-label>
          </q-item-section>
        </q-item>
        <q-item to="/about" clickable>
          <q-item-section>
            <q-item-label>About</q-item-label>
          </q-item-section>
        </q-item>
      </q-list>
    </q-drawer>

    <q-page-container>
      <RouterView />
    </q-page-container>
  </q-layout>
</template>

<style lang="sass" scoped>
</style>
