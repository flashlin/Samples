import type { ILocalQueryClient } from "@/apis/LocalQueryClient";
import { useAppState } from "@/stores/appState";

export class BindWorker {
  run(localQueryClient: ILocalQueryClient) {
    const timerId = setInterval(async () => {
      const appState = useAppState();
      const bindResp = await localQueryClient.knockAsync({
        uniqueId: appState.guid,
        appUid: appState.appUid,
        port: appState.appPort,
      });
      if (!bindResp.isSuccess) {
        clearInterval(timerId);
      }
    }, 1000);
  }
}
