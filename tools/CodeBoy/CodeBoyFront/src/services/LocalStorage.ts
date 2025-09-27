/**
 * LocalStorage service for persisting data
 */
export class LocalStorageService {
  /**
   * Load data from localStorage and deserialize it
   * @param name Storage key name
   * @returns Promise with deserialized data
   */
  static async loadFromStorage<T>(name: string): Promise<T | null> {
    try {
      const data = localStorage.getItem(name)
      if (data === null) {
        return null
      }
      return JSON.parse(data) as T
    } catch (error) {
      console.error(`Failed to load from storage: ${name}`, error)
      return null
    }
  }

  /**
   * Serialize object and save to localStorage
   * @param name Storage key name
   * @param obj Object to serialize and save
   */
  static async saveToStorage<T>(name: string, obj: T): Promise<void> {
    try {
      const serialized = JSON.stringify(obj)
      localStorage.setItem(name, serialized)
    } catch (error) {
      console.error(`Failed to save to storage: ${name}`, error)
      throw error
    }
  }
}
