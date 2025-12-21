import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import JsonEditor from '@/components/JsonEditor.vue'
import type { JsonSchemaField } from '@/components/JsonEditor.vue'

describe('JsonEditor - extractSchema method', () => {
  describe('Null and empty inputs', () => {
    it('should return {} for null input', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const result = wrapper.vm.extractSchema(null)
      expect(result).toEqual({})
    })

    it('should return {} for empty string', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const result = wrapper.vm.extractSchema('')
      expect(result).toEqual({})
    })

    it('should return {} for whitespace-only string', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const result = wrapper.vm.extractSchema('   ')
      expect(result).toEqual({})
    })

    it('should return [] for empty array', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const result = wrapper.vm.extractSchema('[]')
      expect(result).toEqual([])
    })
  })

  describe('Array mode - union of properties', () => {
    it('should extract schema from single object array', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify([{ id: 1, name: 'Alice', age: 30 }])
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(3)
      expect(result).toContainEqual({ key: 'id', type: 'number' })
      expect(result).toContainEqual({ key: 'name', type: 'string' })
      expect(result).toContainEqual({ key: 'age', type: 'number' })
    })

    it('should extract schema from multiple objects with same properties', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify([
        { id: 1, name: 'Alice' },
        { id: 2, name: 'Bob' }
      ])
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(2)
      expect(result).toContainEqual({ key: 'id', type: 'number' })
      expect(result).toContainEqual({ key: 'name', type: 'string' })
    })

    it('should extract union of all properties from incomplete objects', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify([
        { id: 1, name: 'Alice', email: 'alice@test.com' },
        { id: 2, name: 'Bob' },
        { id: 3, age: 25 }
      ])
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(4)
      expect(result).toContainEqual({ key: 'id', type: 'number' })
      expect(result).toContainEqual({ key: 'name', type: 'string' })
      expect(result).toContainEqual({ key: 'email', type: 'string' })
      expect(result).toContainEqual({ key: 'age', type: 'number' })
    })

    it('should detect date type in array', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify([
        { id: 1, createdAt: '2024-01-15T10:30:00Z' }
      ])
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(2)
      expect(result).toContainEqual({ key: 'id', type: 'number' })
      expect(result).toContainEqual({ key: 'createdAt', type: 'date' })
    })

    it('should use majority vote for type conflicts', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify([
        { id: 1, value: 100 },
        { id: 2, value: 'text' },
        { id: 3, value: 'hello' }
      ])
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(2)
      expect(result).toContainEqual({ key: 'id', type: 'number' })
      expect(result).toContainEqual({ key: 'value', type: 'string' })
    })
  })

  describe('Object mode', () => {
    it('should extract schema from simple object', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify({ id: 1, name: 'Alice', age: 30 })
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(3)
      expect(result).toContainEqual({ key: 'id', type: 'number' })
      expect(result).toContainEqual({ key: 'name', type: 'string' })
      expect(result).toContainEqual({ key: 'age', type: 'number' })
    })

    it('should detect date type in object', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify({
        userId: 123,
        lastLogin: '2024-12-21T08:00:00Z'
      })
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toHaveLength(2)
      expect(result).toContainEqual({ key: 'userId', type: 'number' })
      expect(result).toContainEqual({ key: 'lastLogin', type: 'date' })
    })
  })

  describe('Error handling', () => {
    it('should return {} for invalid JSON', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const result = wrapper.vm.extractSchema('invalid json{')
      expect(result).toEqual({})
    })
  })

  describe('Date detection', () => {
    it('should detect ISO 8601 date with timezone', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify({
        date1: '2024-01-15T10:30:00Z',
        date2: '2024-01-15T10:30:00+08:00'
      })
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toContainEqual({ key: 'date1', type: 'date' })
      expect(result).toContainEqual({ key: 'date2', type: 'date' })
    })

    it('should detect simple date format', () => {
      const wrapper = mount(JsonEditor, {
        props: {
          modelValue: null,
          schema: []
        }
      })

      const jsonStr = JSON.stringify({ simpleDate: '2024-01-15' })
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toContainEqual({ key: 'simpleDate', type: 'date' })
    })
  })
})
