<script setup lang="ts">
import { reactive, inject, Ref } from 'vue';
import { CodeSnippet, useCodeSnippetService } from '../models';
import InputText from 'primevue/inputtext';
import InputNumber from 'primevue/inputnumber';
import Button from 'primevue/button';
import Textarea from 'primevue/textarea';
import { DynamicDialogInstance } from 'primevue/dynamicdialogoptions';

const codeSnippetService = useCodeSnippetService();
const dialogRef: Ref<DynamicDialogInstance> = inject("dialogRef")!;

// interface IAddCodeSnippetProps {
//    data: CodeSnippet;
// }
// const props = defineProps<IAddCodeSnippetProps>();

const injectData: CodeSnippet = dialogRef.value.data;

const data = reactive<CodeSnippet>({
   id: injectData.id,
   programLanguage: injectData.programLanguage,
   content: injectData.content,
   description: injectData.description,
});


function onClickAdd() {
   codeSnippetService.upsertCodeAsync(data);   
   dialogRef.value.close(data);
}

</script>

<template>
   <div class="card">
      <div class="p-fluid grid">
         <div class="field col-12">&nbsp;</div>
         <div class="field col-12 md:col-4">
            <span class="p-float-label">
               <InputNumber v-model="data.id" readonly />
               <label for="inputtext">id</label>
            </span>
         </div>
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
