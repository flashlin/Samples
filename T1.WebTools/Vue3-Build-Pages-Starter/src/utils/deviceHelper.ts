export class DeviceHelper {
  public static isMobile(): boolean {
    return /android|mobile/i.test(window.navigator.userAgent)
  }

  public static isOnApp(): boolean {
    return /isonapp=true/i.test(window.location.search)
  }
}
