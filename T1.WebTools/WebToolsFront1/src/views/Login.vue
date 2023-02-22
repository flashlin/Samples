<script setup lang="ts">
import router from '@/router';
import { reactive } from 'vue';
import { useAppState } from '@/stores/appState';
import localQueryHostClient from '@/apis/LocalQueryHostClient';
import localQueryClient from '@/apis/LocalQueryClient';
import { v4 as uuidv4 } from 'uuid';
import { BindWorker } from "@/apis/BindWorker";

const data = reactive({
   username: "",
   password: "",
})
const appState = useAppState();
const login = async () => {
   const guidString: string = uuidv4();
   appState.guid = guidString;
   const unbindLocalQueryAppsInfo = await localQueryHostClient.getUnbindLocalQueryAppsAsync();
   const foundInfo = unbindLocalQueryAppsInfo.find(async info => {
      try {
         var resp = await localQueryClient.knockAsync({
            uniqueId: guidString,
            appUid: info.appUid,
            port: info.port,
         });

         if (resp.isSuccess) {
            appState.appUid = info.appUid;
            appState.appPort = info.port;
            const bindWorker = new BindWorker();
            bindWorker.run();
            appState.localQueryClient = localQueryClient;
         }

         return resp.isSuccess;
      } catch {
         return false;
      }
   });

   if (foundInfo == null) {
      //Login FAIL
      return;
   }

   appState.isAuthenticated = true;
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