<script setup lang="ts">
import {
  getTrainDataPage,
  addTrainData,
  updateTrainData,
  type ITrainDataItem
} from '@/apis/chatTrainDataApi';
import { ref } from 'vue';

interface IData {
  items: ITrainDataItem[];
  editItem: ITrainDataItem;
  editing: boolean;
}

const data = ref<IData>({
  items: [],
  editing: false,
  editItem: {
    id: 0,
    instruction: '',
    input: '',
    output: ''
  }
});

const clickSaveTrainData = (id: number) => {
  const item = data.value.items.find((x) => x.id === id)!;
  updateTrainData(item);
};

const clickAddTrainData = async () => {
  await addTrainData(data.value.editItem);
  data.value.editing = false;
  await fetchData();
};

const clickAdd = () => {
  data.value.editing = true;
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
    <n-button @click="clickAdd()">Add</n-button>
    <n-space vertical v-if="data.editing">
      Instruction
      <n-input type="textarea" v-model="data.editItem.instruction" :maxlength="512" :rows="3"/>
      Input
      <n-input type="textarea" v-model:value="data.editItem.input" :maxlength="512" :rows="3"/>
      Output
      <n-input type="textarea" v-model:value="data.editItem.output" :maxlength="512" :rows="5"/>
      <n-space>
        <n-button @click="clickAddTrainData()">Save</n-button>
      </n-space>
    </n-space>
    <template v-for="item in data.items" :key="item.id">
      <n-space vertical>
        {{ item.id }}
        Instruction
        <n-input
          type="textarea"
          v-model:value="item.instruction"
          :maxlength="512" :rows="3"
        />
        Input
        <n-input type="textarea" v-model:value="item.input" :maxlength="512" :rows="3"/>
        Output
        <n-input type="textarea" v-model:value="item.output" :autosize="{ minRows: 10 }" :maxlength="512" :rows="3" />
        <n-space>
          <n-button @click="clickSaveTrainData(item.id)">Save</n-button>
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
