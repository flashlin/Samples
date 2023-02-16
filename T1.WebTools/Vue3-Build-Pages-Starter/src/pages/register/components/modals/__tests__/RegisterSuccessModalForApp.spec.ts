import { beforeEach, describe, expect, it } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import RegisterSuccessModalForApp from "@/pages/register/components/modals/RegisterSuccessModalForApp.vue"
import { useEntryProps } from '../../../../../composable/useDatasetProps'
import { createTestingPinia } from '@pinia/testing'
import { useUserStore } from '../../../stories/userStore'

vi.mock('@/composable/useDatasetProps')
describe('RegisterSuccessModalForApp tests', () => {
  let app: VueWrapper
  beforeEach(() => {
    vi.mocked(useEntryProps).mockReturnValue({
      brand: 'test_brand',
    })
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout'],
    })
    app = createApp()
  })

  function createApp() {
    return mount(RegisterSuccessModalForApp, {
      global: {
        plugins: [createTestingPinia()],
      },
    })
  }

  const getLoginButton = () => app.findByTestId('login-app')

  it('should execution login function when user click login app', async() => {
    await getLoginButton().trigger('click')

    expect(useUserStore().gotoAppAsiAsync).toHaveBeenCalled()
  })

  it('should disabled button when login in progress', async() => {
    useUserStore().isSubmitting = true
    await app.vm.$nextTick()

    expect((getLoginButton().element as HTMLButtonElement).disabled).toBeTruthy()
  })

  it('should auto trigger login after 10 seconds', async() => {
    vi.advanceTimersByTime(10 * 1000)

    await app.vm.$nextTick()

    expect(useUserStore().gotoAppAsiAsync).toHaveBeenCalled()
  })
})
