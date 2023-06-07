<script setup lang="ts">
import { getImageForClassifier } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';
import { blobToImageUrlData } from 'ts-standard';

let imageBlob = reactive<Blob>(new Blob());
let imageUrl = ref<string>("");

onMounted(async () => {
  imageBlob = await getImageForClassifier();
  const imageData = await blobToImageUrlData(imageBlob);
  imageUrl.value = imageData;
  console.log('url', imageUrl.value);
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
      <button class="btn">Submit</button>
    </div>
  </form>
</template>
