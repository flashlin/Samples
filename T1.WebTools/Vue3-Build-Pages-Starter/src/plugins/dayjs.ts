import dayjs from 'dayjs'
import customParseFormat from 'dayjs/plugin/customParseFormat'
import isSameOrAfter from 'dayjs/plugin/isSameOrAfter'

dayjs.extend(customParseFormat)
dayjs.extend(isSameOrAfter)
export default dayjs