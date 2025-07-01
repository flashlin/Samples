<script lang="ts" setup>
import ComboDropbox from '@/components/ComboDropbox.vue';
import { ref } from 'vue';

const list = ref([
    { label: 'Apple', value: 'apple 1' },
    { label: 'Banana', value: 'banana 2' },
    { label: 'Cherry', value: 'cherry 3' },
    { label: 'Cherryiess', value: 'cherry 4' },
])
const selected = ref('Cherry');
</script>
<template>
    <ComboDropbox :list="list" v-model="selected" />
</template>