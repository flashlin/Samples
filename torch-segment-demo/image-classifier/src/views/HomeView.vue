<script setup lang="ts">
import { getClassifyCategories, getImageForClassifier, sendClassifyImage } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';
import type { ILabel } from '@/models/types';

let imageName = ref<string>('');
let imageUrl = ref<string>("");
let labels = reactive<ILabel[]>([]);

function onClickCategory(id: number) {
  sendClassifyImage(id, imageName.value);
}

onMounted(async () => {
  const resp = await getImageForClassifier();
  imageUrl.value = resp.imageUrl;

  labels.slice(0, labels.length);
  const data = await getClassifyCategories();
  labels.push(...data);
});
</script>

<template>
  <form>
    <div class="form-group">
      <label for="id">Image:</label>
      <img :src="imageUrl">
    </div>
    <div class="form-group">
      <label for="pwd">Label:</label>
      <template v-for="item in labels" :key="item.id">
        <button type='button' class="btn" @click="() => onClickCategory(item.id)">{{ item.label }}</button>
      </template>
    </div>
  </form>
</template>
