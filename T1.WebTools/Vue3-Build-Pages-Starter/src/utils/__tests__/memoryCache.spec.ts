import { describe, expect, it } from 'vitest'
import { CacheEntry, MemoryCache } from '../memoryCache'

describe('memory cache tests', () => {
  const NEVER_EXPIRED_TIMESTAMP = Date.now() + Number.MAX_SAFE_INTEGER
  const TEST_KEY = 'testKey'
  let memoryCache: MemoryCache
  const mockApi = {
    getAsync: vi.fn(),
  }
  beforeEach(() => {
    mockApi.getAsync.mockRestore()
    memoryCache = new MemoryCache()
  })

  it('should set item in cache store', () => {
    const expectedItem = { key: 'value' }
    const givenKey = 'any_key'

    memoryCache.setOrReplace(givenKey, expectedItem, NEVER_EXPIRED_TIMESTAMP)

    expect(memoryCache.get(givenKey)).toEqual(expectedItem)
  })

  it('should replace exist item', () => {
    const oldItem = { key: 'value' }
    const newExpectedItem = { key: 'new_value' }
    const givenKey = 'any_key'

    memoryCache.setOrReplace(givenKey, oldItem, NEVER_EXPIRED_TIMESTAMP)
    memoryCache.setOrReplace(givenKey, newExpectedItem, NEVER_EXPIRED_TIMESTAMP)

    expect(memoryCache.get(givenKey)).toEqual(newExpectedItem)
  })

  it('should return null when cache item not exist', () => {
    expect(memoryCache.get('any_key')).toBeNull()
  })

  it('should be null when cache item expired', () => {
    const givenBeforeTime = Date.now() - 1
    const key = 'any_key'

    memoryCache.setOrReplace(key, {}, givenBeforeTime)

    expect(memoryCache.get(key)).toBeNull()
  })

  it('should return item when cache item not expired', () => {
    const givenAfterTime = Date.now() + 1
    const key = 'any_key'
    const expectedItem = { key: 'value' }

    memoryCache.setOrReplace(key, expectedItem, givenAfterTime)

    expect(memoryCache.get(key)).toEqual(expectedItem)
  })

  it('should send request when cache not found', async() => {
    const expectedItem = { key: 'value' }
    mockApi.getAsync.mockResolvedValue(expectedItem)
    const item = await memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())

    expect(mockApi.getAsync).toHaveBeenCalled()
    expect(item).toEqual(expectedItem)
  })

  it('should not send request when cache found', async() => {
    const expectedItem = { key: 'value' }
    memoryCache.setOrReplace(TEST_KEY, expectedItem, NEVER_EXPIRED_TIMESTAMP)
    const item = await memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())

    expect(mockApi.getAsync).not.toHaveBeenCalled()
    expect(item).toEqual(expectedItem)
  })

  it('should send request once when key is same', async() => {
    const expectedItem = { key: 'value' }
    mockApi.getAsync.mockResolvedValue(expectedItem)
    const item = await memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const item2 = await memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())

    expect(mockApi.getAsync).toHaveBeenCalledTimes(1)
    expect(item).toEqual(expectedItem)
    expect(item2).toEqual(expectedItem)
  })

  it('should send request once when key is same - Promise all', async() => {
    const expectedResponse = { status: 'success' }
    mockApi.getAsync.mockResolvedValue(expectedResponse)

    const task1 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const task2 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const task3 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())

    const allResponse = await Promise.all([task1, task2, task3])

    expect(mockApi.getAsync).toBeCalledTimes(1)
    allResponse.forEach(( response ) => {
      expect(response).toEqual(expectedResponse)
    })
  })

  it('should only be called once api when key is same - synchronous factory and return type is promise ', async() => {
    const expectedResponse = { status: 'success' }

    mockApi.getAsync.mockResolvedValue(expectedResponse)

    const task1 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, () => new Promise((resolve) => resolve(expectedResponse)))
    const task2 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, () => new Promise((resolve) => resolve(expectedResponse)))
    const task3 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, () => new Promise((resolve) => resolve(expectedResponse)))

    const allResponse = await Promise.all([task1, task2, task3])

    allResponse.forEach(( response ) => {
      expect(response).toEqual(expectedResponse)
    })
  })

  it('should only be called once api when key is same - synchronous factory and return type is none function', async() => {
    const expectedValue = 'any_value'
    const task1 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, () => expectedValue)
    const task2 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, () => expectedValue)
    const task3 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, () => expectedValue)

    const allResponse = await Promise.all([task1, task2, task3])

    allResponse.forEach(( response ) => {
      expect(response).toEqual(expectedValue)
    })
  })

  it('should call api 3 times when api always throw error', async() => {
    mockApi.getAsync.mockRejectedValue({})

    const task1 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const task2 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const task3 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())

    await Promise.allSettled([task1, task2, task3])

    expect(mockApi.getAsync).toBeCalledTimes(3)
  })

  it('should call api 2 times when first time api throw error but sec time success', async() => {
    const expectedResponse = { status: 'success' }

    mockApi.getAsync.mockRejectedValueOnce({})
    mockApi.getAsync.mockResolvedValue(expectedResponse)

    const task1 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const task2 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())
    const task3 = memoryCache.getOrCreateAsync(TEST_KEY, NEVER_EXPIRED_TIMESTAMP, async() => await mockApi.getAsync())

    const allResponse = await Promise.allSettled([task1, task2, task3])

    expect(mockApi.getAsync).toBeCalledTimes(2)
    expect(allResponse[0]).toEqual({ status: 'rejected', reason: {} })
    expect(allResponse[1]).toEqual({ status: 'fulfilled', value: expectedResponse })
    expect(allResponse[2]).toEqual({ status: 'fulfilled', value: expectedResponse })
  })
})


describe('cache entry tests', () => {
  it('should be expired when entry timestamp less than now', () => {
    const now = Date.now()
    const afterTime = Date.now() + 1

    const cacheEntry = new CacheEntry({ key: 'value' }, now)

    expect(cacheEntry.isExpired(afterTime)).toBeTruthy()
  })

  it('should not be expired when entry timestamp more than now', () => {
    const now = Date.now()
    const afterTime = Date.now() + 1

    const cacheEntry = new CacheEntry({ key: 'value' }, afterTime)

    expect(cacheEntry.isExpired(now)).toBeFalsy()
  })

  it('should not be expired when entry timestamp equal now', () => {
    const givenNow = Date.now()
    const cacheEntry = new CacheEntry({ key: 'value' }, givenNow)

    expect(cacheEntry.isExpired(givenNow)).toBeFalsy()
  })
})