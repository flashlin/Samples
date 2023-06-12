import { computed } from "vue";

export const KEY_Flash_AppContext = "_flashsql_appContext";

export function generateUuid() {
  let d = Date.now();
  if (
    typeof performance !== "undefined" &&
    typeof performance.now === "function"
  ) {
    d += performance.now();
  }
  return "xxxxxxxx_xxxx_4xxx_yxxx_xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (d + Math.random() * 16) % 16 | 0;
    d = Math.floor(d / 16);
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}


//import * as ko from "knockout";
export const DIV_TO_EXTENSION_EVENTS = "__flashman_events";
export const DIV_VARIABLES = "__flashman_variables";
export const CHANGE_EVENT_NAME = "change";

export enum FlashEventType 
{
  ClientSendAjaxNetworkContent = "ClientSendAjaxNetworkContent",
  UpdateUserInjectionScriptReq = "UpdateUserInjectionScriptReq",
  LoadUserInjectionSettingListReq = "LoadUserInjectionSettingListReq",
  ClientSendSelectorPath = "ClientSendSelectorPath",
  RequestToSelector = "RequestToSelector",
}

export class EventInfo {
  constructor(data?: Partial<EventInfo>) {
    Object.assign(this, data);
  }
  eventName: string = "";
  eventData: any;
}

export class SelectorPathInfo {
  constructor(data?: Partial<SelectorPathInfo>)
  {
    Object.assign(this, data);
  }
  tagName: string = "";
  xpath: string = "";
  css: string = "";
  hostname: string = "";
  url: string = "";
  urlQuery: string = "";
}

export class ExtensionEvent {
  constructor(initData?: Partial<ExtensionEvent>) {
    Object.assign(this, initData);
  }
  htmlElem?: HTMLDivElement;
  eventName: string = "";
  eventData: any;
}

export type ChromeSendResponse = (data?: any) => void;

export class DateTime {
  private _time: Date;
  constructor(time: Date) {
    this._time = time;
  }

  toStr(): string {
    const time = this._time;
    const y = this.padLeadingZeros(time.getFullYear(), 4);
    const MM = this.padLeadingZeros(time.getMonth() + 1, 2);
    const dd = this.padLeadingZeros(time.getDate(), 2);
    const hh = this.padLeadingZeros(time.getHours(), 2);
    const mm = this.padLeadingZeros(time.getMinutes(), 2);
    const ss = this.padLeadingZeros(time.getSeconds(), 2);
    return `${y}-${MM}-${dd} ${hh}:${mm}:${ss}`;
  }

  private padLeadingZeros(num: number, size: number) {
    let s = num + "";
    while (s.length < size) s = "0" + s;
    return s;
  }
}

export class RequestData {
  constructor(initData?: Partial<RequestData>) {
    Object.assign(this, initData);
  }
  url: string = "";
  requestData: any = undefined;
}

export class RequestContent {
  constructor(initData?: Partial<RequestContent>) {
    Object.assign(this, initData);
  }
  startTime: Date = new Date();
  endTime?: Date = undefined;
  url: string = "";
  requestData: any = undefined;
  responseData: any = undefined;
  status: string = "";
  injectedScriptId?: string = undefined;
  isCollapse: boolean = false;

  durationTime: number = computed(() => {
    if (this.endTime == undefined) {
      return 0;
    }
    const endTimeStr: any = this.endTime;
    const startTimeStr: any = this.startTime;
    const endTime = Date.parse(endTimeStr);
    const startTime = Date.parse(startTimeStr);
    return endTime - startTime;
  }) as any;

  // formattedStartTime: string = computed(() => {
  //   let time = this.startTime;
  //   let fmt = `${time.getFullYear()}-${
  //     time.getMonth() + 1
  //   }-${time.getDate()} ${time.getHours()}:${time.getMinutes()}:${time.getSeconds()}`;
  //   return fmt;
  // }) as any;
}

export class UserInjectionSetting 
{
  constructor(initData?: Partial<UserInjectionSetting>) {
    Object.assign(this, initData);
    this.uuid = generateUuid();
  }
  uuid: string = "";
  matchUrlPattern: string = "";
  responseScriptCode: string = "";
  result: any = null;
  enabled: boolean = false;
}