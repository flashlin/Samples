import { describe, expect, it } from 'vitest'
import { DeviceHelper } from '../deviceHelper'

describe('deviceHelper tests', () => {
  it.each([
    ['isOnApp=true'],
    ['isonapp=true'],
    ['isonapp=TRUE'],
    ['ISONAPP=TRUE'],
  ])('should be true when query is %s', ( query: string ) => {
    location.search = `?${ query }`
    expect(DeviceHelper.isOnApp()).toBeTruthy()
  })
  it.each([
    ['isOnApp=false'],
    ['isonapp=FASLE'],
    ['isonappppp=TRUE'],
    ['isOnApp=TTRUE'],
  ])('should be false when query is %s', ( query: string ) => {
    location.search = `?${ query }`
    expect(DeviceHelper.isOnApp()).toBeFalsy()
  })
})
