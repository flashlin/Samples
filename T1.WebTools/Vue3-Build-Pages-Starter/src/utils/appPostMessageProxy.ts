export interface IAppPostMessageProxy {
  post( event: string ): void
}

export class AppPostMessageProxy implements IAppPostMessageProxy {
  post( event: string ): void {
    if (!import.meta.env.PROD) console.log(event)
    if (typeof window.flutterChannel !== 'undefined') {
      window.flutterChannel.postMessage(event)
    }
  }
}
