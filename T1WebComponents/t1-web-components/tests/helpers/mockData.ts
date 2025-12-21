import type { JsonSchemaField } from '@/components/JsonEditor.vue'

export const BASIC_SCHEMA: JsonSchemaField[] = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const }
]

export const FULL_SCHEMA: JsonSchemaField[] = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const },
  { key: 'createdAt', label: 'Created At', type: 'date' as const }
]

export const SINGLE_FIELD_SCHEMA: JsonSchemaField[] = [
  { key: 'value', label: 'Value', type: 'string' as const }
]

export function createArrayJson(items: Record<string, any>[]): string {
  return JSON.stringify(items)
}

export function createObjectJson(data: Record<string, any>): string {
  return JSON.stringify(data)
}

export function createCompactJson(data: any): string {
  return JSON.stringify(data)
}

export function createFormattedJson(data: any): string {
  return JSON.stringify(data, null, 2)
}

export const SAMPLE_ARRAY_DATA = [
  { id: 1, name: 'Alice' },
  { id: 2, name: 'Bob' },
  { id: 3, name: 'Charlie' }
]

export const SAMPLE_OBJECT_DATA = {
  id: 100,
  name: 'Test User',
  createdAt: '2025-01-15'
}
