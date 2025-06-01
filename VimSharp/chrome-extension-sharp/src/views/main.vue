<script lang="ts" setup>
import CardGrid from '@/components/CardGrid.vue'
import vueIcon from '@/assets/vue.svg'

const cards = [
  { icon: vueIcon, title: 'Excel Query', path: '/excelQuery' },
  { icon: vueIcon, title: '功能二', path: '/excel' },
  { icon: vueIcon, title: '功能三', path: '/vim' },
  { icon: vueIcon, title: '功能四', path: '/fileupload' },
]
</script>
<template>
  <CardGrid :cards="cards" />
</template>