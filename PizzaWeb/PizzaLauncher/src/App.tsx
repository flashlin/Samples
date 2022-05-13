import { defineComponent, ref } from "vue";

export default defineComponent({
   props: {},
   setup(props) {

      return () => (
         <div>
            <span>Hello world!</span>
         </div>
      );
   },
});
