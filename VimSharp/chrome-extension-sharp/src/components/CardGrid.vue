<script lang="ts" setup>
import { goTo } from '@/tools/visual-router'

interface CardItem {
  icon: string // icon 路徑
  title: string
  path: string
}

const props = defineProps<{ cards: CardItem[] }>()

function handleCardClick(card: CardItem) {
  if (card.path) {
    goTo(card.path)
  }
}
</script>

<template>
  <div class="w-full min-h-screen bg-gray-900 flex items-center justify-center py-8">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-6 w-full max-w-4xl">
      <div
        v-for="card in props.cards"
        :key="card.title"
        class="flex flex-col items-center bg-gray-800 rounded-xl shadow-lg p-4 cursor-pointer hover:bg-gray-700 transition border border-gray-700"
        @click="handleCardClick(card)"
      >
        <img :src="card.icon" alt="icon" class="w-16 h-16 mb-2 object-contain grayscale" />
        <span class="text-white text-base font-semibold mt-1">{{ card.title }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 卡片 hover 效果與暗黑主題已用 tailwind 實現 */
</style> 