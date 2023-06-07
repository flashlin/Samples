<script setup lang="ts">
import { getImageForClassifier } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';

let imageBlob = reactive<Blob>(new Blob());
let imageUrl = ref<string>("");

function blobToImageData(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const data = reader.result as string;
      resolve(data);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

onMounted(async () => {
  imageBlob = await getImageForClassifier();
  const imageData = await blobToImageData(imageBlob);
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
