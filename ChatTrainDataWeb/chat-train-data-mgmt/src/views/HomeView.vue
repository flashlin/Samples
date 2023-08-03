<script setup lang="ts">
import { getTrainDataPage, updateTrainData, type ITrainDataItem } from '@/apis/chatTrainDataApi';
import { ref } from 'vue';

interface IData {
  items: ITrainDataItem[];
}

const data = ref<IData>({
  items: []
});

const saveTrainData = (id: number) => {
  const item = data.value.items.find((x) => x.id === id)!;
  updateTrainData(item);
};

const fetchData = async () => {
  const res = await getTrainDataPage({
    startIndex: 0,
    pageSize: 20
  });
  data.value.items = res.items;
};

fetchData();
</script>

<template>
  <n-card title="Chat Train Data">
    <template v-for="item in data.items" :key="item.id">
      <n-space vertical>
        {{ item.id }}
        Instruction
        <n-mention type="textarea" v-model:value="item.instruction" />
        Input
        <n-mention type="textarea" v-model:value="item.input" />
        Output
        <n-mention type="textarea" v-model:value="item.output" />
        <n-space>
          <n-button @click="saveTrainData(item.id)">Save</n-button>
        </n-space>
      </n-space>
    </template>
  </n-card>
</template>

<style scoped>
.n-card {
  max-width: 100%;
}
</style>
