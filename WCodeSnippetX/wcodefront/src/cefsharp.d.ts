declare interface IBoundObject {
  queryCode(text: string): Promise<string>;
  setClipboard(text: string): Promise<void>;
  minimize(): Promise<void>;
  bringMeToFront(): Promise<void>;
  upsertCode(codeSnippetJson: string): Promise<void>;
}

declare interface Window {
  __backend: IBoundObject;
}

declare let CefSharp: any;
