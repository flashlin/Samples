<template>
  <div>
    <Toast />
    <ConfirmDialog></ConfirmDialog>
    Vue3 Sample
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