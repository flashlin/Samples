<script lang="ts" setup>
import CardGrid from '@/components/CardGrid.vue'
import vueIcon from '@/assets/vue.svg'

const cards = [
  { icon: vueIcon, title: 'Excel Query', path: 'excelQuery' },
  { icon: vueIcon, title: 'Convert', path: 'convert' },
  { icon: vueIcon, title: 'Csv', path: 'csv' },
  { icon: vueIcon, title: 'TableDesigner', path: 'tableDesigner' },
  { icon: vueIcon, title: 'ScorePrediction', path: 'mgmtScorePrediction' },
  { icon: vueIcon, title: 'Artemis', path: 'artemis' },
]
</script>
<template>
  Support Tools
  <CardGrid :cards="cards" />
</template>