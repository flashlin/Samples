//type ICaseInsensitiveMap<T, U> = Map<T, U>;
export class CaseInsensitiveMap<T, U> {
  constructor(entries?: Array<[T, U]> | Iterable<[T, U]>, ...args) {
    return Reflect.construct(Map, args, CaseInsensitiveMap);
  }

  set(key: T, value: U): this {
    if (typeof key === "string") {
      key = key.toLowerCase() as any as T;
    }
    return Map.prototype.set.call(this, key, value) as this;
  }

  get(key: T): U | undefined {
    if (typeof key === "string") {
      key = key.toLowerCase() as any as T;
    }
    return Map.prototype.get.call(this, key) as U;
  }

  has(key: T): boolean {
    if (typeof key === "string") {
      key = key.toLowerCase() as any as T;
    }
    return Map.prototype.has.call(this, key) as boolean;
  }
}
