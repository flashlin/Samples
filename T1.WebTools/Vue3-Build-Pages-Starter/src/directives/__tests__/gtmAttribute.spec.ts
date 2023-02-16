import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { gtmAttributeDirective } from '../gtmAttribute'
import { defineComponent, ref } from 'vue'

const createTestComponent = ( template: string, categoryName: string, setup?: any ) => {
  return mount(defineComponent({
    template,
    setup,
  }), {
    global: {
      directives: {
        'gtm-attr': gtmAttributeDirective({ categoryName }),
      },
    },
  })
}
describe('gtmAttribute tests', () => {
  const TEST_CATEGORY = 'Test_Category'
  it('gtm label and category should exist and value must be correct and event name first char is uppercase', () => {
    const wrapper = createTestComponent(`<div v-gtm-attr:click="'Test_Element'"></div>`, TEST_CATEGORY)

    expect(wrapper.attributes('data-gtm-category')).toBe(TEST_CATEGORY)
    expect(wrapper.attributes('data-gtm-label')).toMatchInlineSnapshot('"Test_Category_Click_Test_Element"')
  })

  it('should update label value when arg updated', async() => {
    const wrapper = createTestComponent(`<div v-gtm-attr:click="testValue" @click="handleClickUpdateValue"></div>`, TEST_CATEGORY, () => {
      const testValue = ref<string>('first_value')
      const handleClickUpdateValue = () => testValue.value = 'second_value'
      return {
        testValue,
        handleClickUpdateValue,
      }
    })

    await wrapper.trigger('click')

    expect(wrapper.attributes('data-gtm-label')).toBe('Test_Category_Click_second_value')
  })
})
