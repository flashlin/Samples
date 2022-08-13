<script setup lang="ts">
import { reactive } from 'vue';
import { CodeSnippet } from '../models';
import InputText from 'primevue/inputtext';
import Button from 'primevue/button';
import Textarea from 'primevue/textarea';

const data = reactive<CodeSnippet>({
   id: 0,
   programLanguage: '',
   content: '',
   description: '',
});


function onClickAdd() {
   console.log('onClickAdd');
}

</script>

<template>
   <div class="card">
      <div class="p-fluid grid">
         <div class="field col-12">&nbsp;</div>
         <div class="field col-12 md:col-4">
            <span class="p-float-label">
               <InputText type="text" v-model="data.programLanguage" />
               <label for="inputtext">language</label>
            </span>
         </div>
         <div class="field col-12">&nbsp;</div>
         <div class="field col-12">
            <span class="p-float-label">
               <Textarea v-model="data.content" :autoResize="true" rows="5" cols="30" />
               <label for="inputtext">content</label>
            </span>
         </div>
         <div class="field col-12">&nbsp;</div>
         <div class="field col-12">
            <span class="p-float-label">
               <InputText type="text" v-model="data.description" />
               <label for="inputtext">description</label>
            </span>
         </div>
         <div class="field col-12">&nbsp;</div>
         <div class="field col-12">
            <Button label="Add" class="p-button p-component p-button-icon-only" @click="onClickAdd">
               Add
            </Button>
         </div>
      </div>
   </div>
</template>

<style scoped lang="scss">
</style>
