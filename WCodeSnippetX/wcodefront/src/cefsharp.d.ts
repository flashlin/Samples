declare interface IBoundObject {
  queryCodeAsync(text: string): Promise<string>;
  setClipboardAsync(text: string): Promise<void>;
}

declare interface Window {
  __backend: IBoundObject;
}

declare let CefSharp: any;
