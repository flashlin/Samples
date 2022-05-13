import { defineComponent, ref } from "vue";

export default defineComponent({
   props: {},
   setup(props) {
      const canvasRef = ref<HTMLCanvasElement>();

      return () => (
         <div>
            <span>Hello world!</span>
            <canvas ref={canvasRef} />
         </div>
      );
   },
});
