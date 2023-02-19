<script setup lang="ts">
import { reactive } from 'vue';
const data = reactive({
   username: "",
   password: "",
})
const login = () => { }
const clear = () => { }
</script>
<template>
   <q-page class="login-page q-pa-md d-flex flex-center">
      <q-card class="q-mx-auto" style="max-width: 400px;">
         <q-card-section class="text-h6 q-mb-md bg-grey-7">Login</q-card-section>
         <q-form>
            <q-input outlined v-model="data.username" label="Username" class="q-mb-md" />
            <q-input outlined v-model="data.password" label="Password" type="password" class="q-mb-md" />
            <q-card-actions>
               <q-btn type="submit" color="primary" label="Login" class="q-mr-md" />
               <q-btn type="reset" color="secondary" label="Clear" />
            </q-card-actions>
         </q-form>
      </q-card>
   </q-page>
</template>