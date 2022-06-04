import mitt from "mitt";
export const emitter = mitt();

export function toastInfo(message: string) {
  emitter.emit("toast", {
    severity: "info",
    summary: "Info Message",
    detail: message,
    life: 3000,
  });
}

export function toastError(message: string) {
  emitter.emit("toast", {
    severity: "error",
    summary: "Error Message",
    detail: message,
    life: 3000,
  });
}

export interface IConfirm {
  message: string;
  resolve: () => void;
  reject: () => void;
}

export function confirmPopupAsync(message: string) {
  return new Promise<boolean>((resolve, reject) => {
    emitter.emit("confirm", {
      message: message,
      resolve: async () => {
        resolve(true);
      },
      reject: () => {
        reject(false);
      },
    });
  });
}
