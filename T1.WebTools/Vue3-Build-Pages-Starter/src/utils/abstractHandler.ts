interface IAbstractHandler<T> {
  handle( dto: T ): void
}
export abstract class AbstractHandler<T> implements IAbstractHandler<T> {
  constructor( public nextHandler: IAbstractHandler<T> | null ) {
  }

  abstract handle( dto: T ): void
}