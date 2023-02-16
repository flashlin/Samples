<script lang="ts" setup>
interface BaseModalProps {
  title: string
  modalType: 'success' | 'error'
  textClass?: string
}

withDefaults(defineProps<BaseModalProps>(), {
  textClass: '',
})
</script>

<template>
  <div v-bind="$attrs" class="t-modal">
    <div class="t-modal-mask"></div>
    <div class="t-modal-dialog">
      <div class="t-modal-content">
        <div class="t-modal-header">
          <div :class="['icon',`icon-${modalType}`]"></div>
        </div>
        <div class="t-modal-body">
          <div class="t-modal-title">{{ title }}</div>
          <div class="t-modal-text" :class="textClass">
            <slot name="text"></slot>
          </div>
        </div>
        <div class="t-modal-footer">
          <slot name="footer"></slot>
        </div>
      </div>
    </div>
  </div>
</template>
