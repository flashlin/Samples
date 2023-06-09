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
  <table>
    <tr>
      <td>
        <div style="height: 50pt;">
          <label>Image: {{ imageName }}</label>
        </div>
      </td>
    </tr>
    <tr>
      <td>
          <img :src="imageUrl">
      </td>
    </tr>
    <tr>
      <td>
        <div style="height: 50pt;"></div>
      </td>
    </tr>
    <tr>
      <td>
        <div class="form-group">
          <label for="pwd">Label:</label>
          <template v-for="item in labels" :key="item.id">
            <button type='button' class="btn" @click="onClickCategory(item.id)">{{ item.label }}</button>
          </template>
        </div>
      </td>
    </tr>
    <tr>
      <td>
      </td>
    </tr>
  </table>
</template>
