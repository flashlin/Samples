import dayjs, { Dayjs } from "dayjs"
import utc from "dayjs/plugin/utc"
import isToday from 'dayjs/plugin/isToday'
import timezone from "dayjs/plugin/timezone"
import localizedFormat from "dayjs/plugin/localizedFormat"
import buddhistEra  from 'dayjs/plugin/buddhistEra'

import "dayjs/locale/id"
import "dayjs/locale/ja"
import "dayjs/locale/ko"
import "dayjs/locale/my"
import "dayjs/locale/th"
import "dayjs/locale/vi"
import "dayjs/locale/zh"

dayjs.extend(utc)
dayjs.extend(timezone)
dayjs.extend(isToday)
dayjs.extend(localizedFormat)
dayjs.extend(buddhistEra)




const toServerTime = (dayjs: Dayjs) => {
  return dayjs.tz('America/Anguilla')
}
export const getServerTime = ( date?: string) => {
  if(date){
    return toServerTime( dayjs(date) )
  }
  return toServerTime(dayjs())
}

export default dayjs