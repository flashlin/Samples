interface CancelListener {
  cancel(): void
}
type EventCategory = 'register' | 'promotion'
type EventType = 'country-changed' | 'register-success' | 'register-failed' | 'open-login-modal'

interface IEventHelper {
  emit<T>(eventType: EventType, detail?: T): void
  on<T>(eventType: EventType, callback: (detail: T) => void, useCapture?: boolean): CancelListener
}

export class EventHelper implements IEventHelper {
  constructor( private prefix: EventCategory ) {
    this._prefix = prefix
  }
  
  private _prefix: EventCategory
  private addPrefix = (eventType: EventType) => `${ this._prefix }:${ eventType }`

  emit<T>(eventType: EventType, detail?: T) {
    const typeName = this.addPrefix(eventType)
    const customEvent = new CustomEvent<T>(typeName, {
      detail,
    })
    window.dispatchEvent(customEvent)
  }

  on<T>(eventType: EventType, callback: (detail: T) => void, useCapture?: boolean): CancelListener {
    const typeName = this.addPrefix(eventType)

    const listenerCallBack = (event: Event) => {
      callback((event as CustomEvent<T>)?.detail)
    }

    window.addEventListener(typeName, listenerCallBack, useCapture)
    return {
      cancel: () => {
        window.removeEventListener(typeName, listenerCallBack, useCapture)
      },
    }
  }

}
export const useEventHelper = (prefix: EventCategory): IEventHelper => new EventHelper(prefix)