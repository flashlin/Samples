import { ref } from 'vue'

export function useCountdownTimer( fn: () => void, options: { seconds: number } ) {
  let timerId: ReturnType<typeof setTimeout> | null = null
  const seconds = ref(options.seconds)
  const hasExecuted = ref(false)
  const cancel = () => {
    if (timerId) {
      clearTimeout(timerId)
    }
  }

  const execute = () => {
    cancel()
    hasExecuted.value = true
    fn()
  }

  const setupTimer = () => {
    if (seconds.value === 0) {
      execute()
    }
    else {
      timerId = setTimeout(() => {
        seconds.value--
        setupTimer()
      }, 1000)
    }
  }

  setupTimer()

  return {
    seconds,
    cancel,
    execute,
    hasExecuted,
  }
}