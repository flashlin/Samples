<script setup lang="ts">
import router from '@/router';
import { nextTick, reactive } from 'vue';
import { useAppStore } from '@/stores/appStore.js';
import useLocalQueryHostClient from '@/apis/LocalQueryHostClient';
import useLocalQueryClient from '@/apis/LocalQueryClient';
import { v4 as uuidv4 } from 'uuid';
import { BindWorker } from "@/apis/BindWorker";
import { storeToRefs } from 'pinia';

const data = reactive({
   username: "",
   password: "",
})
const appStore = useAppStore();
const login = async () => {
   const localQueryClient = useLocalQueryClient();
   const localQueryHostClient = useLocalQueryHostClient();
   const guidString: string = uuidv4();
   appStore.$patch({ guid: guidString });
   const unbindLocalQueryAppsInfo = await localQueryHostClient.getUnbindLocalQueryAppsAsync();
   let foundInfo = null;
   for await (const info of unbindLocalQueryAppsInfo){
      try {
         localQueryClient.setConnectOption({
            appUid: info.appUid,
            appPort: info.port,
         });
         var resp = await localQueryClient.knockAsync(guidString);
         if (resp.isSuccess) {
            appStore.$patch({ 
               appUid: info.appUid,
               appPort: info.port,
            });
            const bindWorker = new BindWorker();
            bindWorker.run(guidString, localQueryClient);
         }
         if( resp.isSuccess ) {
            foundInfo = info;
            break;
         }
      } catch {
      }
   }

   if (foundInfo == null) {
      //Login FAIL
      return;
   }

   appStore.$patch({ isAuthenticated: true });
   router.push('/');
}
const clear = () => { }
</script>
<template>
   <q-page class="login-page q-pa-md d-flex flex-center">
      <q-card class="q-mx-auto" style="max-width: 400px;">
         <q-card-section class="text-h6 q-mb-md bg-grey-7">Login</q-card-section>
         <div>
            <q-input outlined v-model="data.username" label="Username" class="q-mb-md" />
            <q-input outlined v-model="data.password" label="Password" type="password" class="q-mb-md" />
            <q-card-actions>
               <q-btn type="button" color="primary" label="Login" class="q-mr-md" @click="login" />
               <q-btn type="reset" color="secondary" label="Clear" />
            </q-card-actions>
         </div>
      </q-card>
   </q-page>
</template>