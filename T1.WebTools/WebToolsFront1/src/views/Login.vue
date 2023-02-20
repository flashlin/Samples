<script setup lang="ts">
import router from '@/router';
import { reactive } from 'vue';
import { useAppState } from '@/stores/appState';
import localQueryHostClient from '@/apis/LocalQueryHostClient';
import localQueryClient from '@/apis/LocalQueryClient';

const data = reactive({
   username: "",
   password: "",
})
const appState = useAppState();
const login = async () => {
   const unbindLocalQueryAppsInfo = await localQueryHostClient.getUnbindLocalQueryAppsAsync();
   unbindLocalQueryAppsInfo.forEach(info => {
      localQueryClient.knockAsync({
         uniqueId: "",
      })
   });
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