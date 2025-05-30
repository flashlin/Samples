<template>
  <div style="height: 300px;">
    <RecycleScroller class="scroller" :items="list" :item-size="32" key-field="id" v-slot="{ item }">
      <div class="user">
        <span v-for="(value, key) in item" :key="key" style="margin-right: 8px;">
          {{ value }}
        </span>
      </div>
    </RecycleScroller>
  </div>

</template>

<script lang="ts" setup>
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import { RecycleScroller } from 'vue-virtual-scroller'

// Define props using TypeScript
defineProps<{
  list: Array<{
    id: string | number
    name: string
  }>
}>()
</script>

<style scoped>
.scroller {
  height: 100%;
}

.user {
  height: 32%;
  padding: 0 12px;
  display: flex;
  align-items: center;
}
</style>