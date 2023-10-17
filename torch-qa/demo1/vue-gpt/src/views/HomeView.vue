<template>
  <div class="flex flex-col h-screen">
    <div
      class="flex flex-nowrap fixed w-full items-baseline top-0 px-6 py-4 bg-gray-100"
    >
      <div class="text-2xl font-bold">SupportGPT</div>
      <div class="ml-4 text-sm text-gray-500">
        2023
      </div>
      <div
        class="ml-auto px-3 py-2 text-sm cursor-pointer hover:bg-white rounded-md"
        @click="clickConfig()"
      >
        Settings
      </div>
    </div>


    <ChatInput />
  </div>
</template>

<script setup lang="ts">
import ChatInput from '@/components/ChatInput.vue';
import { ChatGpt } from '@/libs/gpt';

const clickConfig = () => {
};

const gpt = new ChatGpt();
gpt.getLastConversationMessages()
  .then(resp => {
    console.log(resp);
  });
</script>

<style scoped>
pre {
  font-family: -apple-system, "Noto Sans", "Helvetica Neue", Helvetica,
    "Nimbus Sans L", Arial, "Liberation Sans", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN",
    "Microsoft YaHei", "Wenquanyi Micro Hei", "WenQuanYi Zen Hei", "ST Heiti",
    SimHei, "WenQuanYi Zen Hei Sharp", sans-serif;
}
</style>