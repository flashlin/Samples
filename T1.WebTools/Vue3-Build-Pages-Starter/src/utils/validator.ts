const isNullOrUndefined =(value: unknown) => value === undefined || value === null

export const required = ( value: string ): boolean => {
  if (isNullOrUndefined(value)) {
    return false
  }
  return value.trim().length > 0
}
