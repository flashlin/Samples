import type { App, DirectiveBinding } from 'vue'
import { inject } from 'vue'

const SIMPLE_GTM_INJECTION_KEY = Symbol()

class ClickListener {
  elementToCallBack = new WeakMap<HTMLElement, ( e: MouseEvent ) => void>()

  on( el: HTMLElement, cb: ( evt: MouseEvent ) => void ) {
    this.elementToCallBack.set(el, cb)
    el.addEventListener('click', cb, {
      capture: false,
    })
  }

  off( el: HTMLElement ) {
    const cb = this.elementToCallBack.get(el)
    if (cb) {
      el.removeEventListener('click', cb)
      this.elementToCallBack.delete(el)
    }
  }
}

class VisibilityListener {
  observer: IntersectionObserver | undefined
  elementToCallBack = new WeakMap<Element, () => void>()

  constructor() {
    if ('IntersectionObserver' in window) {
      this.createObserver()
    }
    else {
      // try again after polyfill loaded
      setTimeout(() => {
        this.createObserver()
      }, 1000)
    }
  }

  private createObserver(): void {
    if ('IntersectionObserver' in window) {
      this.observer = new IntersectionObserver(( entries, observer ) => {
        entries.forEach(( entry ) => {
          if (entry.isIntersecting) {
            const el = entry.target
            const cb = this.elementToCallBack.get(el)
            if (cb) cb()
            this.elementToCallBack.delete(el)
            observer.unobserve(entry.target)
          }
        })
      }, {})
    }
  }

  on( el: HTMLElement, cb: () => void ) {
    this.elementToCallBack.set(el, cb)
    if (this.observer) this.observer.observe(el)
  }

  off( el: HTMLElement ) {
    const cb = this.elementToCallBack.get(el)
    if (cb) this.elementToCallBack.delete(el)

    if (this.observer) this.observer.unobserve(el)
  }
}

class DurationCalculator {
  constructor( private startTime: number = performance.now() ) {
  }

  getDuration(): number {
    const endTime = performance.now()
    return Math.round((endTime - this.startTime) / 1000)
  }
}

class GtmDirective {
  private _clickListener: ClickListener
  private _visibilityListener: VisibilityListener

  constructor( private simpleGtm: SimpleGtm ) {
    this._clickListener = new ClickListener()
    this._visibilityListener = new VisibilityListener()
  }

  bindEvent( el: HTMLElement, binding: DirectiveBinding<string | Record<string, string>> ) {
    switch (binding.arg) {
      case 'click':
        this._clickListener.on(el, ( evt: MouseEvent ) => {
          evt.stopPropagation()
          this.simpleGtm.pushEvent('click', this.getPostData(binding.value))
        })
        break
      case 'view':
        this._visibilityListener.on(el, () => {
          this.simpleGtm.pushEvent('view', this.getPostData(binding.value))
        })
        break
      case 'duration':
        this.createDurationEventOnce(el)
        break
    }
  }

  private createDurationEventOnce( el: HTMLElement ) {
    const duration = new DurationCalculator()
    this._clickListener.on(el, ( evt ) => {
      evt.stopPropagation()
      this.simpleGtm.pushEvent('duration', this.getPostData(duration.getDuration().toString()))
      this._clickListener.off(el)
    })
  }

  private getPostData( data: string | Record<string, string> ): Record<string, string> {
    return typeof data === 'string' ? { label: data } : data
  }

  unmountedEvent( el: HTMLElement, binding: DirectiveBinding ) {
    switch (binding.arg) {
      case 'click':
        this._clickListener.off(el)
        break
      case 'view':
        this._visibilityListener.off(el)
        break
      case 'duration':
        this._clickListener.off(el)
    }
  }
}

export interface ISimpleGtm {
  pushEvent( action: string, data: Record<string, string> ): void
}

class SimpleGtm implements ISimpleGtm {
  private GTM_ID_PATTERN = /^GTM-[0-9A-Z]+$/

  constructor( private gtmOptions: GtmOptions ) {
    if (gtmOptions.gtmCode) {
      this.loadGtmScript(gtmOptions.gtmCode)
    }
  }

  private loadGtmScript( gtmCode: string ) {
    if (!this.GTM_ID_PATTERN.test(gtmCode)) {
      console.debug(`Invalid GTM ID: ${ gtmCode } --- load GTM script was canceled`)
      return
    }

    if (this.hasSameGtmId()) return
    const scriptTag = document.createElement('script')
    window.dataLayer = window.dataLayer || []
    window.dataLayer.push({
      "event": "gtm.js",
      "gtm.start": new Date().getTime(),
    })

    scriptTag.async = true
    scriptTag.src = `https://www.googletagmanager.com/gtm.js?id=${ gtmCode }`
    document.head.appendChild(scriptTag)
  }

  pushEvent( action: string, data: Record<string, string> ) {
    if (this.gtmOptions.enabled) {
      window.dataLayer?.push({
        action,
        category: this.gtmOptions.category,
        ...this.gtmOptions.extendData,
        ...data,
      })
    }
  }

  private hasSameGtmId() {
    return Array.from(document.getElementsByTagName("script")).some(( script ) =>
      script.src.includes(`googletagmanager.com/gtm.js?id=${ this.gtmOptions.gtmCode }`),
    )
  }
}

interface GtmOptions {
  enabled: boolean
  category: string
  gtmCode?: string
  extendData?: Record<string, string>
}


export const createSimpleGtm = ( options: GtmOptions ) => {
  const simpleGtm = new SimpleGtm(options)
  return ( app: App ) => {
    const gtmDirective = new GtmDirective(simpleGtm)
    app.provide<SimpleGtm>(SIMPLE_GTM_INJECTION_KEY, simpleGtm)
    app.directive('gtm-event', {
      mounted: ( el, binding ) => gtmDirective.bindEvent(el, binding),
      unmounted: ( el, binding ) => gtmDirective.unmountedEvent(el, binding),
    })
  }
}

export const useSimpleGtm = (): ISimpleGtm => {
  const instance = inject<ISimpleGtm>(SIMPLE_GTM_INJECTION_KEY)
  if (!instance) throw new Error('useSimpleGtm must be used after createSimpleGtm')
  return instance
}