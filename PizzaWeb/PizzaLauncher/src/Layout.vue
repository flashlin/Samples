<template>
  <div>
    <Toast />
    <ConfirmDialog></ConfirmDialog>
    <Menubar :model="state.items">
      <template #start>
        <img alt="logo" src="https://www.primefaces.org/wp-content/uploads/2020/05/placeholder.png" height="40"
          class="mr-2">
      </template>
      <template #item="{ item }">
        &nbsp;
        <span @click="handleClickMenuItem(item)">{{ item.label }}</span>
      </template>
      <template #end>
        <InputText placeholder="Search" type="text" />
      </template>
    </Menubar>
    <router-view></router-view>
  </div>
</template>

<script setup lang="ts">
import { defineComponent, onMounted, reactive, ref } from "vue";
import Toast, { ToastMessageOptions } from "primevue/toast";
import { useToast } from "primevue/usetoast";
import { emitter, IConfirm } from "@/models/AppToast";
import ConfirmPopup from "primevue/confirmpopup";
import ConfirmDialog from 'primevue/confirmdialog';
import { useConfirm } from "primevue/useconfirm";
import Menubar from 'primevue/menubar';
import { MenuItem } from "primevue/menuitem";
import router from "./Router";
import InputText from "primevue/inputtext";

const state = reactive({
  items: [
    {
      label: 'Template',
      key: '/',
      icon: 'pi pi-fw pi-file',
    },
    {
      label: 'Banner',
      key: '/banners',
      icon: 'pi pi-fw pi-file',
    }
  ],
});

function handleClickMenuItem(item: MenuItem)
{
  router.push(item.key!);
}

onMounted(() => {
  const toast = useToast();
  emitter.on("toast", (data) => {
    toast.add(data as ToastMessageOptions);
  });

  const confirm = useConfirm();
  emitter.on("confirm", (arg) => {
    console.log('1111');
    let data = arg as IConfirm;
    confirm.require({
      message: data.message,
      header: 'Confirmation',
      icon: "pi pi-exclamation-triangle",
      defaultFocus: "reject",
      acceptClass: "p-button-text",
      accept: () => {
        data.resolve();
      },
      reject: () => {
        data.reject();
      },
    });
  });
});
</script>

<style lang="scss">
</style>