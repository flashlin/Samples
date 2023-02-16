import { reactive, toRef } from 'vue'
import { useInterval } from '@/composable/useInterval'
import { Time } from '@/utils/time'

export interface Interval {
  remove: () => void
  start: (_ms?: number | undefined) => NodeJS.Timeout | undefined
}
export const useTimer =( expiry: number, autoStart= true) => {
  let interval: Interval
  const DEFAULT_DELAY = 1000
  function getDelayFromExpiryTimestamp(expiryTimestamp: number) {
    const seconds = Time.getSecondsFromExpiry(expiryTimestamp)
    const extraMilliSeconds = Math.floor((seconds - Math.floor(seconds)) * 1000)
    return extraMilliSeconds > 0 ? extraMilliSeconds : DEFAULT_DELAY
  }
  const state = reactive<{
    expiryTimestamp: number
    seconds: number
    isRunning: boolean
    isExpired: boolean
    delay: null | number
    didStart: boolean
  }>({
    expiryTimestamp: expiry,
    seconds: Time.getSecondsFromExpiry(expiry),
    isRunning: autoStart,
    isExpired: false,
    delay: getDelayFromExpiryTimestamp(expiry),
    didStart: autoStart,
  })

  function _handleExpire() {
    state.isExpired = true
    state.isRunning = false
    state.delay = null
    if (interval) interval.remove()
  }

  function pause() {
    state.isRunning = false
    if (interval) interval.remove()
  }

  function restart(newExpiryTimestamp: number = expiry, newAutoStart = true) {
    pause()
    state.delay = getDelayFromExpiryTimestamp(newExpiryTimestamp)
    state.didStart = newAutoStart
    state.isExpired = false
    state.expiryTimestamp = newExpiryTimestamp
    state.seconds = Time.getSecondsFromExpiry(newExpiryTimestamp)
    if (state.didStart) start()
  }

  function resume() {
    const time = new Date()
    const newExpiryTimestamp = time.setMilliseconds(
      time.getMilliseconds() + state.seconds * 1000,
    )
    restart(newExpiryTimestamp)
  }

  function start() {
    if (state.didStart) {
      state.seconds = Time.getSecondsFromExpiry(state.expiryTimestamp)
      state.isRunning = true
      interval = useInterval(
        () => {
          if (state.delay !== DEFAULT_DELAY) {
            state.delay = DEFAULT_DELAY
          }
          const secondsValue = Time.getSecondsFromExpiry(state.expiryTimestamp)
          state.seconds = secondsValue
          if (secondsValue <= 0) {
            _handleExpire()
          }
        },
        state.isRunning ? state.delay : null,
      )
    }
    else {
      resume()
    }
  }
  restart(expiry, autoStart)
  return {
    ...Time.getTimeFromSeconds(toRef(state, 'seconds')),
    start,
    pause,
    resume,
    restart,
    isRunning: toRef(state, 'isRunning'),
    isExpired: toRef(state, 'isExpired'),
  }
}