<script setup lang="ts">
import { getImageForClassifier } from '@/models/api';
import { onMounted, reactive, ref } from 'vue';
import type { ILabel } from '@/models/types';

let imageName = ref<string>('');
let imageUrl = ref<string>("");
let fileRef = ref<HTMLInputElement>();

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

function drawleave(event: Event) {
  event.preventDefault();
  //dropzone.style.backgroundColor = "";
  console.log('leave');
}

function drag(event: DragEvent) {
  console.log('drag');
  const dataTransfer = event.dataTransfer as DataTransfer;
  var file = dataTransfer.files[0];
  console.log(file);
}

function handleFileUpload() {
  console.log('upload')
  const files: FileList = fileRef.value?.files!;
  const file = files[0];
  console.log(file)
}

onMounted(async () => {
});
</script>

<template>
  <div class="mt-5">
    <div class="row">
      <div class="col-md-6 offset-md-3">
        <div class="dropzone" @dragover="drawover" @drag="drag($event)" @click="selectFile">
          <h4>Drag and Drop or Click to Upload Banner</h4>
        </div>
        <input ref="fileRef" type="file" class="d-none" @change="handleFileUpload" hidden>
        <div class="text-center mt-3">
          <button class="btn btn-primary">Upload</button>
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
