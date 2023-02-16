export interface IValidator {
  isValid( value: unknown ): boolean | Promise<boolean>
}

export interface IPromiseValidator {
  isValid(value: unknown): Promise<boolean>
}