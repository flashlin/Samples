<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { EditorRender, VisualEditor } from './EditorBuffer';
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
    const editor = new VisualEditor();
    const render = new EditorRender(editor);
    render.createCanvas(canvas);
    //const ctx = canvas.getContext('2d')!;
    //ctx.fillStyle = 'dark';
    //ctx.fillRect(0, 0, props.width, props.height);
}
function initialize() {

}
</script>

<template>
    <div ref="canvasRef"></div>
</template>
