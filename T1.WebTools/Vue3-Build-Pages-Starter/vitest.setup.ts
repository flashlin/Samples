import type { VueWrapper } from "@vue/test-utils"
import { config } from "@vue/test-utils"
import { vi } from 'vitest'
import { ErrorMessage, Field, Form } from 'vee-validate'
import type { DOMWrapper } from '@vue/test-utils/dist/domWrapper'
import { defineComponent } from 'vue'

const DataTestIdPlugin = ( wrapper: VueWrapper ) => {
  function getSelector( testId: string ) {
    return `[data-testid='${ testId }']`
  }

  function findByTestId<T extends Element = Element>( selector: string ) {
    return wrapper.find<T>(getSelector(selector))
  }

  function getByTestId<T extends Element = Element>( testId: string ): Omit<DOMWrapper<T>, 'exists'> {
    return wrapper.get<T>(getSelector(testId))
  }

  return {
    findByTestId,
    getByTestId,
  }
}

vi.stubGlobal('location', new URL('http://test.com'))
vi.stubGlobal('open', vi.fn())

const mockI18nT = defineComponent({
  template: `
    <div v-for="slot of slots" v-bind="$attrs">
    <slot :name="slot"></slot>
    </div>
  `,
  setup( _, context ) {
    return {
      slots: Object.keys(context.slots),
    }
  },
})

config.plugins.VueWrapper.install(DataTestIdPlugin)
config.global.stubs = {
  Form,
  i18nT: mockI18nT,
  Field,
  ErrorMessage,
}
config.global.directives = {
  GtmAttr: () => vi.fn(),
  GtmEvent: () => vi.fn(),
}
config.global.mocks = {
  $t: vi.fn().mockReturnValue(''),
}
