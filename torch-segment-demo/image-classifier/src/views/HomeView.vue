<script setup lang="ts">
import { getClassifyCategories, getImageForClassifier } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';
import { blobToImageUrlData } from 'ts-standard';
import type { ILabel } from '@/models/types';

let imageName = ref<string>('');
let imageBlob = reactive<Blob>(new Blob());
let imageUrl = ref<string>("");
let labels = reactive<ILabel[]>([]);

function onClickCategory(id: number) {
  sendClassifyImage(id, imageName)
}

onMounted(async () => {
  const resp = await getImageForClassifier();
  imageBlob = resp.image;
  const imageData = await blobToImageUrlData(imageBlob);
  imageUrl.value = imageData;

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
        <button type='button' class="btn">{{ item.label }}</button>
      </template>
    </div>
  </form>
</template>
