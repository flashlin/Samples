import { describe } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import RegisterSuccessModal from '@/pages/register/components/modals/RegisterSuccessModal.vue'
import { createTestingPinia } from '@pinia/testing'
import { useUserStore } from '../../../stories/userStore'
import { useEntryProps } from '../../../../../composable/useDatasetProps'

vi.mock('../../../../../composable/useDatasetProps')
describe('RegisterSuccessModal tests', () => {
  const MOCK_LANGUAGE = 'EN'
  let app: VueWrapper
  beforeEach(() => {
    vi.mocked(useEntryProps).mockReturnValue({ brand: 'test', language: MOCK_LANGUAGE })
    app = createTestWrapper()
  })

  function createTestWrapper() {
    return mount(RegisterSuccessModal, {
      global: {
        plugins: [
          createTestingPinia(),
        ],
      },
    })
  }

  it('should process goto payment when user click goto deposit button', () => {
    getGotoDepositButton().trigger('click')

    expect(useUserStore().gotoPlaySiteAsync).toHaveBeenCalledWith({
      language: MOCK_LANGUAGE,
      product: 'payment',
    })
  })

  it('should process goto landing when user click goto landing button', () => {
    const gotoLandingButton = app.findByTestId('gotoLandingButton')
    gotoLandingButton.trigger('click')

    expect(useUserStore().gotoPlaySiteAsync).toHaveBeenCalledWith({
      language: MOCK_LANGUAGE,
      product: 'landing',
    })
  })

  function getGotoDepositButton() {
    return app.findByTestId('gotoDepositButton')
  }

  it('should disabled goto deposit button when user is submitting', async() => {
    useUserStore().isSubmitting = true
    await app.vm.$nextTick()

    expect((getGotoDepositButton().element as HTMLButtonElement).disabled).toBeTruthy()
  })
})