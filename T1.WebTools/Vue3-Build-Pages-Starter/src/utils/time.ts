import { computed, type Ref } from 'vue'
import dayjs from '@/utils/dayjs'

export class Time {
  static getSecondsFromExpiry( expiry: number, shouldRound?: boolean ): number {
    const milliSecondsDistance = expiry - dayjs().valueOf()
    if (milliSecondsDistance > 0) {
      const val = milliSecondsDistance / 1000
      return shouldRound ? Math.round(val) : val
    }
    return 0
  }

  static getTimeFromSeconds( secs: Ref<number> ) {
    const totalSeconds = computed(() => Math.ceil(secs.value))
    const days = computed(() => Math.floor(totalSeconds.value / (60 * 60 * 24)))
    const hours = computed(() => Math.floor((totalSeconds.value % (60 * 60 * 24)) / (60 * 60)))
    const minutes = computed(() => Math.floor((totalSeconds.value % (60 * 60)) / 60))
    const seconds = computed(() => Math.floor(totalSeconds.value % 60))
    return {
      seconds,
      minutes,
      hours,
      days,
    }
  }

  static getGMTOffsetString() {
    const offset = dayjs().utcOffset()
    const offsetOperator = offset >= 0 ? '+' : '-'
    const offsetHour = ( Math.abs(offset) /60 ).toString()
    return `${offsetOperator}${offsetHour}`
  }
}