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

async function onClickCategory(id: number) {
  await sendClassifyImage(id, imageName.value);
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
  <div class="mt-5">
    <div class="row">
      <div class="col-md-6 offset-md-3">
        <div class="dropzone" id="dropzone">
          <h4>Drag and Drop or Click to Upload Banner</h4>
        </div>
        <div class="text-center mt-3">
          <button class="btn btn-primary" id="uploadBtn">Upload</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dropzone {
  border: 2px dashed #ccc;
  padding: 20px;
  text-align: center;
}

.dropzone:hover {
  background-color: #f1f1f1;
}
</style>
