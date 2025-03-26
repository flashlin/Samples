// Chrome API 類型聲明
declare namespace chrome {
  namespace runtime {
    function getURL(path: string): string;
    function getManifest(): any;
    
    const onInstalled: {
      addListener(callback: (details: {
        reason: 'install' | 'update' | 'chrome_update' | 'shared_module_update';
        previousVersion?: string;
        id?: string;
      }) => void): void;
    };
    
    const onMessage: {
      addListener(
        callback: (
          message: any,
          sender: {
            tab?: {
              id: number;
              url: string;
            };
            frameId?: number;
            id?: string;
            url?: string;
            origin?: string;
          },
          sendResponse: (response?: any) => void
        ) => boolean | void
      ): void;
    };
    
    function sendMessage(
      extensionId: string | null,
      message: any,
      options?: { includeTlsChannelId?: boolean },
      callback?: (response: any) => void
    ): void;
    
    function sendMessage(
      message: any,
      callback?: (response: any) => void
    ): void;
  }
  
  namespace storage {
    interface StorageArea {
      get(
        keys: string | string[] | { [key: string]: any } | null,
        callback?: (items: { [key: string]: any }) => void
      ): void;
      
      set(
        items: { [key: string]: any },
        callback?: () => void
      ): void;
    }
    
    const local: StorageArea;
    const sync: StorageArea;
    
    const onChanged: {
      addListener(
        callback: (
          changes: { [key: string]: { oldValue?: any; newValue?: any } },
          areaName: string
        ) => void
      ): void;
    };
  }
  
  namespace tabs {
    interface Tab {
      id?: number;
      index: number;
      windowId: number;
      highlighted: boolean;
      active: boolean;
      pinned: boolean;
      url?: string;
      title?: string;
      favIconUrl?: string;
      status?: string;
      incognito: boolean;
      width?: number;
      height?: number;
      sessionId?: string;
    }
    
    function query(
      queryInfo: {
        active?: boolean;
        currentWindow?: boolean;
        highlighted?: boolean;
        status?: string;
        title?: string;
        url?: string | string[];
        windowId?: number;
        windowType?: string;
        index?: number;
        pinned?: boolean;
        audible?: boolean;
        muted?: boolean;
        groupId?: number;
      },
      callback: (result: Tab[]) => void
    ): void;
    
    function create(
      createProperties: {
        windowId?: number;
        index?: number;
        url?: string;
        active?: boolean;
        pinned?: boolean;
        openerTabId?: number;
      },
      callback?: (tab: Tab) => void
    ): void;
    
    function sendMessage(
      tabId: number,
      message: any,
      options?: { frameId?: number },
      callback?: (response: any) => void
    ): void;
  }
} 