import type { App } from 'vue'
import { getCurrentInstance, inject } from 'vue'

const EVENT_BUS_INJECTION_KEY = Symbol()
export interface IEventHub<T extends string> {
  dispose(): void
  $on( type: T, handler: Function ): void
  $off( type: T, handler: Function ): void
  $once( type: T, handler: Function ): void
  $emit( type: T, ...args: any[] ): void
}

class EventHub<T extends string> implements IEventHub<T> {
  private $_all: Map<string, Array<Function>>
  constructor() {
    this.$_all = new Map()
  }

  dispose() {
    this.$_all.clear()
  }


  $on( type: string, handler: Function ) {
    const handlers = this.$_all.get(type)
    const added = handlers && handlers.push(handler)

    if (!added) {
      this.$_all.set(type, [handler])
    }
  }

  $off( type: string, handler: Function ) {
    const handlers = this.$_all.get(type) || []

    const newHandlers = handler ? handlers.filter(( x ) => x !== handler) : []

    if (newHandlers.length) {
      this.$_all.set(type, newHandlers)
    }
    else {
      this.$_all.delete(type)
    }
  }

  $once( type: string, handler: Function ) {
    const wrapHandler = ( ...args: any[] ) => {
      this.$off(type, wrapHandler)
      handler(...args)
    }
    this.$on(type, wrapHandler)
  }

  $emit( type: string, ...args: any[] ) {
    const handlers = this.$_all.get(type) || []

    handlers.forEach(( handler ) => {
      handler(...args)
    })
  }
}

export const createEventBus = <T extends string>() => ({
  install: ( app: App ) => {
    app.provide<IEventHub<T>>(EVENT_BUS_INJECTION_KEY, new EventHub<T>())
  },
})

export const useEventBus = <T extends string>(): IEventHub<T> => {
  const vueInstance = getCurrentInstance()
  if(!vueInstance){
    throw new Error('cannot be used outside the vue component')
  }
  const EventBus = inject<IEventHub<T>>(EVENT_BUS_INJECTION_KEY)
  if(!EventBus){
    throw new Error('did not register event bus')
  }

  return EventBus
}
