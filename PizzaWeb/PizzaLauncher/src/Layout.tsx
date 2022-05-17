import { defineComponent, onMounted, reactive, ref } from "vue";

export default defineComponent({
  props: {},
  setup(props) {
    return () => (
      <div class="container">
        <router-view></router-view>
      </div>
    );
  },
});
