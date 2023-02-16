import { describe, expect, it } from 'vitest'
import { flushPromises, mount, VueWrapper } from '@vue/test-utils'
import LoginName from "@/components/fields/LoginName.vue"
import { useEventBus } from '@vueuse/core'
import { SetLoginNameUnavailableEvent } from '@/components/fields/fieldEvents'
import { createVeeValidate } from '@/plugins/veeValidate'
import { nextTick } from 'vue'

describe('LoginNameField tests', () => {
  let app: VueWrapper
  beforeEach(() => {
    app = mount(LoginName, {
      attachTo: document.body,
      global: {
        plugins: [
          createVeeValidate(),
        ],
      },
    })
  })

  it('should show tips on first time', async() => {
    await flushPromises()
    expect(getTipListElement().exists()).toBeTruthy()
  })


  it('should hide tips on input dirty', async() => {
    await getLoginNameInput().setValue('test')
    await flushPromises()

    expect(getTipListElement().exists()).toBeFalsy()
  })

  it('should focus input and set error when received SetLoginNameUnavailableEvent event', async() => {
    const mockSuggestions = ['test_loginName']
    const { emit } = useEventBus(SetLoginNameUnavailableEvent)
    emit({ suggestLists: mockSuggestions })

    await flushPromises()
    await nextTick()

    expect(app.findByTestId('suggestions').isVisible()).toBeTruthy()
    focusElementShouldBe(getLoginNameInput().element)
  })

  function focusElementShouldBe( element: Element ) {
    expect(document.activeElement).toEqual(element)
  }

  function getLoginNameInput() {
    return app.find('input[name="loginName"]')
  }

  function getTipListElement() {
    return app.findByTestId('loginNameTips')
  }
})