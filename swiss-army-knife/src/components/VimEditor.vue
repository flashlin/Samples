<script setup lang="ts">
import { onMounted, ref } from 'vue';
const canvasRef = ref<HTMLCanvasElement>();
interface VimEditorProps {
    //modelValue: string;
    width: number;
    height: number;
}
const props = withDefaults(defineProps<VimEditorProps>(), {
    width: 800,
    height: 400,
});
onMounted(() => {
    drawOnCanvas();
});
function drawOnCanvas() {
    const canvas = canvasRef.value!;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'dark';
    ctx.fillRect(0, 0, props.width, props.height);
}
function initialize() {

}
</script>

<template>
    <canvas ref="canvasRef" width="800" height="400"> </canvas>
</template>
