import mitt from 'mitt';
export const emitter = mitt();

export function toastInfo(message: string) {
  emitter.emit('toast', {
    severity: "info",
    summary: "Info Message",
    detail: message,
    life: 3000,
  });
  
}

export function toastError(message: string) {
  emitter.emit('toast', {
    severity: "error",
    summary: "Error Message",
    detail: message,
    life: 3000,
  });
}
