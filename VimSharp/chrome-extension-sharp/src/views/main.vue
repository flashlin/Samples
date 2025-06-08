<script lang="ts" setup>
import CardGrid from '@/components/CardGrid.vue'
import vueIcon from '@/assets/vue.svg'

const cards = [
  { icon: vueIcon, title: 'Excel Query', path: 'excelQuery' },
  { icon: vueIcon, title: 'Convert', path: 'convert' },
  { icon: vueIcon, title: 'ScorePrediction', path: 'mgmtScorePrediction' },
  { icon: vueIcon, title: '功能四', path: 'fileupload' },
]
</script>
<template>
  Support Tools
  <CardGrid :cards="cards" />
</template>