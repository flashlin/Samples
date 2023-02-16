import { isNullOrUndefined } from '@/utils/shared'

export interface IMemoryCache {
  setOrReplace( key: string, item: unknown, absoluteExpiration: number ): void

  get<TItem>( key: string ): TItem | null

  clear(): void

  delete( key: string ): boolean

  getOrCreateAsync<TItem>( key: string, absoluteExpiration: number, factory: () => Promise<TItem> ): Promise<TItem>
}

export class MemoryCache implements IMemoryCache {
  private _entries: Map<string, CacheEntry> = new Map()
  private _pendingSet: Set<string> = new Set()

  setOrReplace( key: string, item: unknown, absoluteExpiration: number ) {
    this._entries.set(key, new CacheEntry(item, absoluteExpiration))
  }

  get<TItem = unknown>( key: string ): TItem | null {
    const cacheEntry = this._entries.get(key)
    if (isNullOrUndefined(cacheEntry)) return null

    if (cacheEntry.isExpired(Date.now())) {
      this._entries.delete(key)
      return null
    }

    return cacheEntry.item as TItem

  }

  clear() {
    this._entries.clear()
  }

  delete( key: string ): boolean {
    return this._entries.delete(key)
  }

  async getOrCreateAsync<TItem = unknown>( key: string, absoluteExpiration: number, factory: () => Promise<TItem> | TItem ): Promise<TItem> {
    if (!this._pendingSet.has(key)) {
      const itemFromCache = this.get(key)

      if (itemFromCache) {
        return itemFromCache as TItem
      }

      try {
        this._pendingSet.add(key)
        const item = await factory()
        this.setOrReplace(key, item, absoluteExpiration)
        return item
      }
      finally {
        this._pendingSet.delete(key)
      }
    }

    return new Promise(( resolve ) => {
      setTimeout(() => {
        resolve(this.getOrCreateAsync(key, absoluteExpiration, factory))
      }, 100)
    })
  }
}

export class CacheEntry {
  constructor( public item: unknown, private _absoluteExpiration: number ) {
  }

  isExpired( timestamp: number ) {
    return this._absoluteExpiration < timestamp
  }
}