export class RefAllocator {
  private nextValue = 1

  next(): string {
    const value = `e${this.nextValue}`
    this.nextValue += 1
    return value
  }
}
