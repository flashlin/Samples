import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount, VueWrapper } from '@vue/test-utils'
import RegisterForm from '../RegisterForm.vue'
import { useEntryProps } from '@/composable/useDatasetProps'
import type { RegisterAppProps } from '../../types'
import { sendRegisterAsync } from '../../api/registerApiClient'
import type { RegisterResponse } from '../../api/types'
import { sendRegistrationSuccessToIncomeAccess } from '../../core/tracking'
import { useEventBus } from '@vueuse/core'
import {
  SetLoginNameUnavailableEvent,
  SetPromotionCodeInvalidEvent,
  TriggerReloadValidationCodeEvent,
  TriggerValidationCodeErrorEvent,
} from '@/components/fields/fieldEvents'
import { useNotification } from '../../../../composable/useNotification'
import { createTestingPinia } from '@pinia/testing'
import { useSimpleGtm } from '../../core/simpleGtm'
import { useRegisterStore } from '../../stories/registerStore'

vi.mock('../../api/registerApiClient')
vi.mock('@/composable/useDatasetProps')
vi.mock('../../../../composable/useNotification')
vi.mock('../../core/tracking')
vi.mock('../../core/simpleGtm')
vi.mock('../../../../utils/shared')

function createTestApp(): VueWrapper {
  return mount(RegisterForm, {
    attachTo: document.body,
    global: {
      stubs: {
        PersonalInfoSection: true,
        BasicInfoSection: true,
        PreferenceInfoSection: true,
        RegisterFailedModalAsync: true,
        RegisterSuccessModalAsync: true,
      },
      plugins: [createTestingPinia()],
    },
  })
}

function mockRegisterResponse( response: RegisterResponse ) {
  vi.mocked(sendRegisterAsync).mockResolvedValue(response)
}

describe('register successfully behavior tests', () => {
  const mockSimpleGtm= {
    pushEvent: vi.fn(),
  }
  beforeEach(() => {
    vi.mocked(useSimpleGtm).mockReturnValue(mockSimpleGtm)
  })
  const getRegisterSuccessModal = ( app: VueWrapper ) => app.findByTestId('registerSuccessModal')


  function givenMockEntryProps( props: Partial<RegisterAppProps> ) {
    vi.mocked(useEntryProps).mockReturnValue(props)
  }

  async function submitForm( app: VueWrapper ) {
    const submitButton = app.findByTestId('submitButton')
    await submitButton.trigger('click')
    await flushPromises()
  }


  it('should send gtm tracking and show register success modal after register successfully and btag not matched income access', async() => {
    givenMockEntryProps({
      isRegisterable: true,
      isBtagMatchIaFormat: false,
    })
    const app = createTestApp()
    useRegisterStore().$patch({
      allowCountries: [{}],
    })
    await app.vm.$nextTick()
    const mockSuccessResponse = {
      success: true,
      isoCurrency: 'test_currency',
      hashedCustId: 'test_hashed_cust_id',
      landingUrl: '',
      isOauth: false,
    }
    mockRegisterResponse(mockSuccessResponse)

    await submitForm(app)

    expect(getRegisterSuccessModal(app).exists()).toBeTruthy()
    expect(sendRegistrationSuccessToIncomeAccess).not.toHaveBeenCalled()
    expect(mockSimpleGtm.pushEvent).toHaveBeenCalledWith('complete', {
      label: 'registration_complete',
    })
  })

  it('should send income access tracking after register successfully when btag is matched income access', async() => {
    givenMockEntryProps({
      isRegisterable: true,
      isBtagMatchIaFormat: true,
    })
    const app = createTestApp()
    useRegisterStore().$patch({
      allowCountries: [{}],
    })
    await app.vm.$nextTick()
    const mockSuccessResponse = {
      success: true,
      isoCurrency: 'test_currency',
      hashedCustId: 'test_hashed_cust_id',
      landingUrl: '',
      isOauth: false,
    }
    mockRegisterResponse(mockSuccessResponse)

    await submitForm(app)

    expect(sendRegistrationSuccessToIncomeAccess).toHaveBeenCalledWith(mockSuccessResponse.hashedCustId)
  })
})

describe('register failed behavior tests', () => {
  const mockNotificationAlert = vi.fn()
  let app: VueWrapper
  beforeEach(() => {
    vi.mocked(useEntryProps).mockReturnValue({
      isRegisterable: true,
    })
    vi.mocked(useNotification).mockReturnValue({
      alert: mockNotificationAlert,
    })
    app = createTestApp()

    useRegisterStore().$patch({
      allowCountries: [{}],
    })
  })
  enableAutoUnmount(afterEach)

  it('should load validation captcha when error response isShowCaptcha is true', async() => {
    const mockReloadValidationCodeEvent = vi.fn()
    useEventBus(TriggerReloadValidationCodeEvent).on(mockReloadValidationCodeEvent)
    mockRegisterResponse({
      success: false,
      errorCode: '',
      suggestions: [],
      isShowCaptcha: true,
    })

    await submitForm()

    expect(mockReloadValidationCodeEvent).toHaveBeenCalled()
    expect(app.findComponent({ name: 'PreferenceInfoSection' }).props('showCaptcha')).toBeTruthy()
  })


  it('should popup registerFailedModal when error starts with E8', async() => {
    const mockReloadValidationCodeEvent = vi.fn()
    const mockRegisterFailedResponse = {
      success: false,
      errorCode: 'E80000',
      suggestions: [],
      isShowCaptcha: false,
    }
    useEventBus(TriggerReloadValidationCodeEvent).on(mockReloadValidationCodeEvent)
    mockRegisterResponse(mockRegisterFailedResponse)

    await submitForm()

    const registerFailedModal = app.findByTestId('registerFailedModal')
    expect(registerFailedModal.exists()).toBeTruthy()
    expect(registerFailedModal.attributes()).toContain({
      errorcode: mockRegisterFailedResponse.errorCode,
    })
  })

  it('should trigger login name unavailable event when error code is E70002', async() => {
    const mockReloadValidationCodeEvent = vi.fn()
    useEventBus(SetLoginNameUnavailableEvent).on(( detail ) => {
      mockReloadValidationCodeEvent(detail)
    })
    const mockRegisterFailedResponse = {
      success: false,
      errorCode: 'E70002',
      suggestions: ['test_loginName'],
      isShowCaptcha: false,
    }
    mockRegisterResponse(mockRegisterFailedResponse)

    await submitForm()

    expect(mockReloadValidationCodeEvent).toHaveBeenCalledWith({ suggestLists: mockRegisterFailedResponse.suggestions })
  })

  it('should trigger login name unavailable event when error code is E700026', async() => {
    const mockTriggerValidationCodeErrorEvent = vi.fn()
    useEventBus(TriggerValidationCodeErrorEvent).on(mockTriggerValidationCodeErrorEvent)
    const mockRegisterFailedResponse = {
      success: false,
      errorCode: 'E700026',
      suggestions: [],
      isShowCaptcha: false,
    }
    mockRegisterResponse(mockRegisterFailedResponse)

    await submitForm()

    expect(mockTriggerValidationCodeErrorEvent).toHaveBeenCalled()
  })

  it.each([
    ['E700022'],
    ['E700023'],
    ['E700024'],
    ['E700025'],
    ['E700039'],
  ])('should emit PromotionCodeInvalidEvent when error code is %s', async( errorCode ) => {
    const mockSetPromotionCodeInvalidEvent = vi.fn()
    useEventBus(SetPromotionCodeInvalidEvent).on(mockSetPromotionCodeInvalidEvent)
    const mockRegisterFailedResponse = {
      success: false,
      errorCode,
      suggestions: [],
      isShowCaptcha: false,
    }
    mockRegisterResponse(mockRegisterFailedResponse)

    await submitForm()

    expect(mockSetPromotionCodeInvalidEvent).toHaveBeenCalled()
  })

  it.each([
    ['E7000'],
    ['E70000'],
    ['E6000'],
  ])('should alert message with error code when error code not in RegisterErrorCode enum', async(errorCode) => {
    const mockRegisterFailedResponse = {
      success: false,
      errorCode: errorCode,
      suggestions: [],
      isShowCaptcha: false,
    }
    mockRegisterResponse(mockRegisterFailedResponse)

    await submitForm()

    expect(mockNotificationAlert).toHaveBeenCalledWith('ContactSboSupportWithErrorCode', { errorCode })
  })

  async function submitForm() {
    const submitButton = app.findByTestId('submitButton')
    await submitButton.trigger('click')
    await flushPromises()
  }
})