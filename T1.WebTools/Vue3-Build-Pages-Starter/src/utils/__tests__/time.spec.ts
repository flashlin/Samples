import { beforeEach, describe, it } from 'vitest'
import { Time } from '../time'
import dayjs from '@/utils/dayjs'

vi.mock('@/utils/dayjs')
describe('Time tests', () => {
  const mock: any = {
    utcOffset: vi.fn(),
  }
  beforeEach(() => {
    vi.mocked(dayjs).mockReturnValue(mock)
  })

  it.each([
    [-480, '-8'],
    [480, '+8'],
    [450, '+7.5'],
    [0, '+0'],
  ])('should get correct format', ( offset: number, expectedDisplay: string ) => {
    mock.utcOffset.mockReturnValue(offset)
    expect(Time.getGMTOffsetString()).toBe(expectedDisplay)
  })
})