import { NDialogProvider, NLoadingBarProvider, NMessageProvider, NNotificationProvider } from "naive-ui";
import { defineComponent, onMounted, reactive, ref } from "vue";

export default defineComponent({
  props: {},
  setup(props) {
    return () => (
      <div class="container">
        <NLoadingBarProvider>
          <NMessageProvider>
            <NNotificationProvider>
              <NDialogProvider>
                <router-view></router-view>
              </NDialogProvider>
            </NNotificationProvider>
          </NMessageProvider>
        </NLoadingBarProvider>
      </div>
    );
  },
});
