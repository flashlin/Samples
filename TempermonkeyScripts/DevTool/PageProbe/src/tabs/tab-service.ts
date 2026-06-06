import { BrowserProtocolError, type TabInfo } from "@page-probe/protocol"

export class TabService {
  async list(): Promise<{ tabs: TabInfo[] }> {
    const tabs = await chrome.tabs.query({})
    return {
      tabs: tabs.flatMap((tab) => {
        if (!tab.id) {
          return []
        }
        return [{
          id: tab.id,
          active: tab.active,
          title: tab.title ?? "",
          url: tab.url ?? "",
          ...(tab.status ? { status: tab.status } : {})
        }]
      })
    }
  }

  async getActive(): Promise<TabInfo> {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    if (!tab?.id) {
      throw new BrowserProtocolError("TAB_NOT_FOUND", "Active tab was not found")
    }
    return {
      id: tab.id,
      active: tab.active,
      title: tab.title ?? "",
      url: tab.url ?? "",
      ...(tab.status ? { status: tab.status } : {})
    }
  }
}
