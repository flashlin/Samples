import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useUserStore } from '../userStore'
import { AuthClient } from '@b2c/sbo-auth'
import { flushPromises } from '@vue/test-utils'
import { LanguageType } from '../../../../constants/language'

const mockAuthClient = {
  processAppLoginAsync: vi.fn(),
  processLoginAsync: vi.fn(),
}
vi.mock('@b2c/sbo-auth', () => {
  return {
    AuthClient: vi.fn().mockImplementation(() => {
      return mockAuthClient
    }),
  }
})
describe('user store tests', () => {
  const MOCK_LOGIN_DTO = {
    landingUrl: 'test_landing_url',
    username: 'test_username',
    password: 'test_password',
  }
  let store: ReturnType<typeof useUserStore>
  beforeEach(() => {
    vi.useFakeTimers()
    mockAuthClient.processAppLoginAsync.mockClear()
    mockAuthClient.processLoginAsync.mockClear()
    setActivePinia(createPinia())
    store = useUserStore()
  })

  function givenMockLoginUser() {
    store.$patch(MOCK_LOGIN_DTO)
  }

  it('should use correct user info and app info dto to process app login', () => {
    givenMockLoginUser()
    const mockAppLoginDto = {
      client_id: 'test_client_id',
      redirect_uri: 'test_redirect_uri',
    }

    store.gotoAppAsiAsync(mockAppLoginDto)

    expect(vi.mocked(AuthClient)).toHaveBeenCalledWith({
      landingDomain: MOCK_LOGIN_DTO.landingUrl,
    })

    expect(mockAuthClient.processAppLoginAsync).toHaveBeenCalledWith({
      username: MOCK_LOGIN_DTO.username,
      password: MOCK_LOGIN_DTO.password,
      ...mockAppLoginDto,
    })
  })


  it('should set isSubmitting ture before process app login and set isSubmitting false after process app login', async() => {
    store.gotoAppAsiAsync({
      client_id: 'any_id',
      redirect_uri: 'any_redirect_uri',
    })

    expect(store.isSubmitting).toBeTruthy()

    await flushPromises()
    vi.advanceTimersByTime(500)

    expect(store.isSubmitting).toBeFalsy()
  })

  it('processAppLoginAsync should not be executed repeatedly when isSubmitting is true', () => {

    store.isSubmitting = true

    store.gotoAppAsiAsync({
      client_id: 'any_id',
      redirect_uri: 'any_redirect_uri',
    })

    expect(mockAuthClient.processAppLoginAsync).not.toHaveBeenCalled()
  })
  it('should use correct user info , product and language to process desktop login', () => {
    givenMockLoginUser()
    const RequestDto: Parameters<typeof store.gotoPlaySiteAsync>[0] = {
      language: LanguageType.EN,
      product: 'payment',
    }

    store.gotoPlaySiteAsync(RequestDto)

    expect(vi.mocked(AuthClient)).toHaveBeenCalledWith({
      landingDomain: MOCK_LOGIN_DTO.landingUrl,
    })
    expect(mockAuthClient.processLoginAsync).toHaveBeenCalledWith({
      username: MOCK_LOGIN_DTO.username,
      password: MOCK_LOGIN_DTO.password,
      ...RequestDto,
    })
  })


  it('should set isSubmitting ture before process user login and set isSubmitting false after process user login', async() => {
    store.gotoPlaySiteAsync({
      language: LanguageType.EN,
      product: 'payment',
    })

    expect(store.isSubmitting).toBeTruthy()

    await flushPromises()
    vi.advanceTimersByTime(500)

    expect(store.isSubmitting).toBeFalsy()
  })

  it('processAppLoginAsync should not be executed repeatedly when isSubmitting is true', () => {
    store.isSubmitting = true

    store.gotoPlaySiteAsync({
      language: LanguageType.EN,
      product: 'payment',
    })

    expect(mockAuthClient.processLoginAsync).not.toHaveBeenCalled()
  })
})
