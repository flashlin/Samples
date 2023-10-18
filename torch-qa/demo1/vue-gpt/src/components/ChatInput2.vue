<template>
   <div class="flex-1 mx-2 mt-20 mb-2" ref="chatListDom">
      <div class="group flex flex-col px-4 py-3 hover:bg-slate-100 rounded-lg"
         v-for="item of messageList.filter((v) => v.role !== 'system')" :key="item.id">
         <div class="flex justify-between items-center mb-2">
            <div class="font-bold">{{ roleAlias[item.role] }}:</div>
            <CopyButton class="invisible group-hover:visible" :content="item.content" />
         </div>
         <div>
            <div class="prose text-sm text-slate-600 leading-relaxed" v-if="item.content" v-html="md.render(item.content)">
            </div>
            <LoadingBar v-else />
         </div>
      </div>
   </div>

   <div class="sticky bottom-0 w-full p-6 pb-8 bg-gray-100">
      <div class="flex">
         <!-- <input class="input" type="text" placeholder="please input your question" v-model="messageContent"
            @keydown.enter="isTalking || sendMessageOnEnter()" /> -->

         <textarea class="input w-full pt-3 bg-transparent chat-input-area form-input placeholder:text-slate-400/70"
            placeholder="Write the messages.." v-model="messageContent" style="height: 450px!important;"
            @keyup.enter="isTalking || sendMessageOnEnter"
            :style="{ 'height': (messageContent.split('\n').length * 1.5) + 'rem', 'max-height': '15rem', 'min-height': '2.8rem' }"
            spellcheck="false"></textarea>
         <button class="btn" :disabled="isTalking" @click="sendMessageOnEnter()">
            Send
         </button>
      </div>
   </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import CopyButton from '@/components/CopyButton.vue';
import LoadingBar from '@/components/LoadingBar.vue';
import type { ChatMessage } from "@/libs/gpt";
import { md } from "@/libs/markdown";
import { ChatGpt } from "@/libs/gpt";

// const props = defineProps<{
//    messageList: ChatMessage[]
// }>()

let isTalking = ref(false);
let messageContent = ref("");
const chatListDom = ref<HTMLDivElement>();
const roleAlias = { user: "ME", assistant: "ChatGPT", system: "System" };
const messageList = ref<ChatMessage[]>([]);
const chatGpt = new ChatGpt();

const appendMessage = (item: ChatMessage) => {
   const oldMessageList = messageList.value;
   oldMessageList.push(item);
   messageList.value = oldMessageList;
   return item;
}

const replaceLastMessage = (item: ChatMessage) => {
   const oldMessageList = messageList.value.slice(0, messageList.value.length - 1);
   oldMessageList.push(item);
   messageList.value = oldMessageList;
   return item;
}

const sendMessageOnEnter = async () => {
   if (!messageContent.value.length) return;
   appendMessage({
      id: messageList.value.length + 1,
      role: 'user',
      content: messageContent.value
   });
   try {
      const lastMessage = appendMessage({
         id: messageList.value.length + 1,
         role: 'assistant',
         content: ''
      })
      const content = messageContent.value;
      messageContent.value = "";
      const response = await chatGpt.ask(content, (typing: string) => {
         lastMessage.content += typing;
         replaceLastMessage(lastMessage);
      });
      replaceLastMessage(response);
   } catch (e: any) {
      replaceLastMessage({
         id: messageList.value.length + 1,
         role: 'system',
         content: e.message
      });
   }
};

chatGpt.getLastConversationMessages()
   .then(resp => {
      messageList.value = resp.messages;
   });
</script>