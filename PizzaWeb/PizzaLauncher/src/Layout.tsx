import {
  NDialogProvider,
  NLoadingBarProvider,
  NMessageProvider,
  NNotificationProvider,
  darkTheme,
  NConfigProvider,
} from "naive-ui";
import { defineComponent, onMounted, reactive, ref } from "vue";

export default defineComponent({
  props: {},
  setup(props) {
    return () => (
      <div class="container">
        <NConfigProvider theme={darkTheme}>
          <NLoadingBarProvider>
            <NMessageProvider>
              <NNotificationProvider>
                <NDialogProvider>
                  <router-view></router-view>
                </NDialogProvider>
              </NNotificationProvider>
            </NMessageProvider>
          </NLoadingBarProvider>
        </NConfigProvider>
      </div>
    );
  },
});
