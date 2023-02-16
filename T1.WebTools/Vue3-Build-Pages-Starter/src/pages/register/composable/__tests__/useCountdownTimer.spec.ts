import { describe, it } from 'vitest'
import { useCountdownTimer } from '../useCountdownTimer'

describe('useCountdownTimer tests', () => {
  let mockFn: () => void
  beforeEach(() => {
    vi.useFakeTimers()
    mockFn = vi.fn()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('should execute fn after 1 seconds', () => {
    const seconds = 1
    useCountdownTimer(mockFn, { seconds })

    vi.advanceTimersByTime(1000 * seconds)

    expect(mockFn).toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('should executed directly when seconds eq 0', () => {
    useCountdownTimer(mockFn, { seconds: 0 })

    expect(mockFn).toHaveBeenCalled()
  })


  it('hasExecuted status should be ture when run execute directly', () => {
    const { execute, hasExecuted } = useCountdownTimer(mockFn, { seconds: 10 })
    expect(hasExecuted.value).toBeFalsy()

    execute()

    expect(mockFn).toHaveBeenCalled()
    expect(hasExecuted.value).toBeTruthy()
  })

  it('should stop timer', () => {
    const seconds = 1
    const { cancel, hasExecuted } = useCountdownTimer(mockFn, { seconds })

    cancel()
    vi.advanceTimersByTime(seconds * 1000)

    expect(mockFn).not.toHaveBeenCalled()
    expect(hasExecuted.value).toBeFalsy()
  })

  it('should update seconds value', () => {
    const seconds = 10
    const { seconds: secondsRef } = useCountdownTimer(mockFn, { seconds })

    expect(secondsRef.value).toEqual(seconds)
    vi.advanceTimersByTime(1000)
    expect(secondsRef.value).toEqual(seconds - 1)
    vi.advanceTimersByTime(1000)
    expect(secondsRef.value).toEqual(seconds - 2)
  })

})
