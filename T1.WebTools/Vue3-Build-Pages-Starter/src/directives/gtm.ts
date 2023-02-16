import type { App, DirectiveBinding } from 'vue'

class GtmTracking {
  constructor(private gtmOptions: GtmOptions) {
  }
  onMounted(el: HTMLElement, binding: DirectiveBinding) {
    const label = binding.value
    el.dataset.gtmCategory = this.gtmOptions.category
    el.dataset.gtmLabel = label
    switch (binding.arg) {
      case 'click':
        el.addEventListener('click', () => {
          window.dataLayer?.push()
        })
        break
    }
  }
}

interface GtmOptions {
  category: string
}
export const useGtmDirective = (options: GtmOptions) => {
  return (app: App) => {
    const gtmTracking = new GtmTracking(options)
    app.directive('gtm', {
      mounted: (el, binding) => gtmTracking.onMounted(el, binding),
    })
  }
}