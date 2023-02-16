export interface onSubmitValidationErrors<T> {
  errors: Partial<Record<keyof T, string>>
}