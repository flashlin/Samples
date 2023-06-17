<script setup lang="ts">
import { getImageForClassifier, sendImageSegmentation } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';
import { base64ToBlob } from 'ts-standard';
import type { IImageSegmentationItem } from '@/models/types';

let imageName = ref<string>('');
let imageUrl = ref<string>("");
let fileRef = ref<HTMLInputElement>();
let segmentations = reactive<IImageSegmentationItem[]>([]);

async function getNewImage() {
  const resp = await getImageForClassifier();
  imageName.value = resp.name;
  imageUrl.value = resp.imageUrl;
}

function selectFile() {
  fileRef.value!.click();
  console.log('click')
}

async function drawover(event: Event) {
  event.preventDefault();
  //dropzone.style.backgroundColor = "#f1f1f1";
  console.log('over');
}

function drag(event: DragEvent) {
  console.log('drag');
  const dataTransfer = event.dataTransfer as DataTransfer;
  var file = dataTransfer.files[0];
  console.log(file);
}

function convertFileToImageSrc(file: File): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      resolve(reader.result as string);
    };
    reader.readAsDataURL(file);
  });
}

async function handleFileUpload() {
  const files: FileList = fileRef.value!.files!;
  const file: File = files[0];
  imageUrl.value = await convertFileToImageSrc(file);
}

async function clickUploadButton() {
  const files: FileList = fileRef.value!.files!;
  const file: File = files[0];
  console.log('send', file);
  const resp = await sendImageSegmentation(file);
  console.log('image', resp);

  resp.shotImages.forEach(shot => {
    const shotBlob = base64ToBlob(shot.image);
    segmentations.push({
      shotImage: URL.createObjectURL(shotBlob),
      maskImage: '',
      label: shot.label
    })
  })
}

onMounted(async () => {
});
</script>

<template>
  <div class="mt-5">
    <div class="row">
      <div class="col-md-7 offset-md-3">
        <div class="dropzone" @dragover="drawover" @drag="drag($event)" @click="selectFile">
          <template v-if="!imageUrl">
            <h4>Drag and Drop or Click to Upload Banner</h4>
          </template>
          <img :src="imageUrl" v-if="imageUrl" />
        </div>
        <input ref="fileRef" type="file" class="d-none" @change="handleFileUpload" hidden>
        <div class="text-center mt-3">
          <button class="btn btn-primary" @click="clickUploadButton">Upload</button>
        </div>
      </div>
      <div v-for="seg in segmentations" :key="seg.shotImage">
        <img :src="seg.shotImage">
        <span>{{ seg.label }}</span>
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
