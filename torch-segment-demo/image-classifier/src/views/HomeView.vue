<script setup lang="ts">
import { getClassifyCategories, getImageForClassifier, sendClassifyImage } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';
import type { ILabel } from '@/models/types';

let imageName = ref<string>('');
let imageUrl = ref<string>("");
let labels = reactive<ILabel[]>([]);

async function getNewImage() {
  const resp = await getImageForClassifier();
  imageName.value = resp.name;
  imageUrl.value = resp.imageUrl;
}

function onClickCategory(id: number) {
  sendClassifyImage(id, imageName.value);
  getNewImage();
}

onMounted(async () => {
  getNewImage();
  labels.slice(0, labels.length);
  const data = await getClassifyCategories();
  labels.push(...data);
});
</script>

<template>
  <form>
    <div class="form-group">
      <label for="id">Image: {{ imageName }}</label>
      <img :src="imageUrl">
    </div>
    <div class="form-group">
      <label for="pwd">Label:</label>
      <template v-for="item in labels" :key="item.id">
        <button type='button' class="btn" @click="onClickCategory(item.id)">{{ item.label }}</button>
      </template>
    </div>
  </form>
</template>
