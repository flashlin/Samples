import type { IAppPostMessageProxy } from '@/utils/appPostMessageProxy'
import { DeviceHelper } from '@/utils/deviceHelper'

export interface IAppEventEmitter {
  tryPost( event: string ): boolean
}


export class AppEventEmitter implements IAppEventEmitter {
  constructor( private eventPrefix: string, private appPostMessageProxy: IAppPostMessageProxy ) {
  }

  tryPost( event: string ): boolean {
    try {
      if (!DeviceHelper.isOnApp()) return false
      this.appPostMessageProxy.post(`${ this.eventPrefix }_${ event }`)
      return true
    }
    catch (e) {
      console.error(e)
      return false
    }
  }
}
