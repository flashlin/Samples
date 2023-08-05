export type AccessorFunc<T, V> = (obj: T) => V;

export class Accessor<T, V> {
  private _obj: T;
  private _accessor: AccessorFunc<T, V>;
  private _key: string;

  constructor(obj: T, accessor: AccessorFunc<T, V>) {
    this._obj = obj;
    this._accessor = accessor;
    this._key = this.getAccessorPropertyName();
  }

  getValue(): V {
    return this._accessor(this._obj);
  }

  /**
   * const a = new Accessor(item, (x) => x.instruction)
   * a.setValue('new value')
   * equals
   * item.instruction = 'new value'
   * @param value new value
   */
  setValue(value: V): void {
    const key = this._key;
    this._obj[key as keyof T] = value as T[keyof T];
  }

  private getAccessorPropertyName(): string {
    const funcStr = this._accessor.toString();
    //(x) => x.name
    const regex1 = new RegExp(/^\([a-zA-Z]\)\s=>\s[a-zA-Z]\.(.*)$/);
    const match1 = regex1.exec(funcStr);
    if (match1 && match1.length >= 2) {
      return match1[1];
    }
    //function (x) { return x.name; }
    const regex2 = new RegExp(/^function\s\([a-zA-Z]\)\s{\sreturn\s[a-zA-Z]\.(.*);\s}$/);
    const match2 = regex2.exec(funcStr);
    if (match2 && match2.length >= 2) {
      return match2[1];
    }
    throw new Error(`Invalid accessor function '${funcStr}'`);
  }
}