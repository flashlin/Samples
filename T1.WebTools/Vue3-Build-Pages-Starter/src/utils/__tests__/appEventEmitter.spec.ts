import { describe, expect, it } from 'vitest'
import { AppEventEmitter } from '@/utils/appEventEmitter'
import { mock, type MockProxy } from 'vitest-mock-extended'
import type { IAppPostMessageProxy } from '@/utils/appPostMessageProxy'
import { DeviceHelper } from '../deviceHelper'

vi.mock('@/utils/deviceHelper')
describe('appEventEmitter tests', () => {
  const TEST_PREFIX = 'test_prefix'
  let mockAppPostMessageProxy: MockProxy<IAppPostMessageProxy>
  let sut: AppEventEmitter
  beforeEach(() => {
    mockAppPostMessageProxy = mock<IAppPostMessageProxy>()
    sut = new AppEventEmitter(TEST_PREFIX, mockAppPostMessageProxy)
  })

  function givenIsOnApp( isOnApp: boolean ) {
    vi.mocked(DeviceHelper.isOnApp).mockReturnValue(isOnApp)
  }

  it('should return false when is isOnApp is false', () => {
    givenIsOnApp(false)

    expect(sut.tryPost('any_event')).toBeFalsy()
  })

  it('should return false when post message throw error', () => {
    givenIsOnApp(true)
    const logError = vi.spyOn(console, 'error')
      .mockImplementationOnce(() => {
      })
    mockAppPostMessageProxy.post.mockImplementation(() => {
      throw new Error('any_error')
    })

    expect(sut.tryPost('any_event')).toBeFalsy()
    expect(logError).toHaveBeenCalled()
  })

  it('should post event with prefix', () => {
    givenIsOnApp(true)

    sut.tryPost('any_event')

    expect(mockAppPostMessageProxy.post).toHaveBeenCalledWith(expect.stringContaining(TEST_PREFIX))
  })
})
