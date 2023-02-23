import type { ILocalQueryClient } from "@/apis/LocalQueryClient";
import { useAppStore } from "@/stores/appStore";

export class BindWorker {
  run(guid: string, localQueryClient: ILocalQueryClient) {
    const timerId = setInterval(async () => {
      const bindResp = await localQueryClient.knockAsync(guid);
      if (!bindResp.isSuccess) {
        clearInterval(timerId);
      }
    }, 1000);
  }
}
