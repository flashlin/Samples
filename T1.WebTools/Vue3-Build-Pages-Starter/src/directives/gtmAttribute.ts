import type { App, Directive, DirectiveBinding } from 'vue'

interface GtmAttributesOptions {
  categoryName: string
}

const firstCharToUpperCase = ( str: string ) => str.charAt(0).toUpperCase() + str.slice(1)

export function gtmAttributeDirective( options: GtmAttributesOptions ): Directive {
  function createLabelValue( arg?: string, name: string = '' ) {
    const action = arg ? `${ firstCharToUpperCase(arg) }_` : ''
    return `${ options.categoryName }_${ action }${ name }`
  }

  const GTM_LABEL_NAME = 'data-gtm-label'
  const GTM_CATEGORY_NAME = 'data-gtm-category'

  return {
    created: ( el: HTMLElement, { arg, value }: DirectiveBinding<string> ) => {
      const label = createLabelValue(arg, value)
      el.setAttribute(GTM_CATEGORY_NAME, options.categoryName)
      el.setAttribute(GTM_LABEL_NAME, label)
    },
    beforeUpdate: ( el: HTMLElement, { arg, value }: DirectiveBinding<string> ) => {
      const currentLabel = el.getAttribute(GTM_LABEL_NAME)
      if (!currentLabel || !currentLabel.includes(value)) {
        const label = createLabelValue(arg, value)
        el.setAttribute(GTM_LABEL_NAME, label)
      }
    },
  }
}

export function installGtmAttributeDirective( options: GtmAttributesOptions ) {
  const directive = gtmAttributeDirective(options)
  return ( app: App ) => {
    app.directive('gtm-attr', directive)
  }
}
