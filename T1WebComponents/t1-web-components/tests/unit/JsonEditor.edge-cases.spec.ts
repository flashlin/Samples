import { describe, it, expect, vi } from 'vitest'
import { nextTick } from 'vue'
import {
  mountJsonEditor,
  clickButton,
  fillInputField,
  getEmittedModelValue,
  clickRowActionButton
} from '../helpers/jsonEditorTestUtils'
import { BASIC_SCHEMA, createArrayJson, createObjectJson } from '../helpers/mockData'

describe('JsonEditor - Edge Cases', () => {
  describe('Invalid JSON Handling', () => {
    it('should display error for malformed JSON', () => {
      const wrapper = mountJsonEditor({
        modelValue: '{invalid: json}',
        schema: BASIC_SCHEMA
      })

      expect(wrapper.text()).toContain('Invalid JSON')
    })

    it('should emit error event for invalid JSON', () => {
      const wrapper = mountJsonEditor({
        modelValue: 'not json at all',
        schema: BASIC_SCHEMA
      })

      expect(wrapper.emitted('error')).toBeTruthy()
    })

    it('should show error message when JSON parse fails', () => {
      const wrapper = mountJsonEditor({
        modelValue: '{"unclosed":',
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('.text-red-400').exists()).toBe(true)
    })

    it('should recover when valid JSON provided after error', async () => {
      const wrapper = mountJsonEditor({
        modelValue: 'invalid',
        schema: BASIC_SCHEMA
      })

      expect(wrapper.text()).toContain('Invalid JSON')

      await wrapper.setProps({
        modelValue: createObjectJson({ id: 1, name: 'Valid' })
      })
      await nextTick()

      expect(wrapper.text()).not.toContain('Invalid JSON')
    })
  })

  describe('Empty Data Handling', () => {
    it('should handle empty array', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(true)
      expect(wrapper.text()).toContain('No matching items found')
    })

    it('should handle empty object', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({}),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.findAll('input').length).toBeGreaterThan(0)
    })

    it('should initialize form from schema when empty string', () => {
      const wrapper = mountJsonEditor({
        modelValue: '',
        schema: BASIC_SCHEMA
      })

      const idInput = wrapper.find('input#form-id') as any
      const nameInput = wrapper.find('input#form-name') as any

      expect(idInput.element.value).toBe('0')
      expect(nameInput.element.value).toBe('')
    })

    it('should handle null modelValue', () => {
      const wrapper = mountJsonEditor({
        modelValue: null,
        schema: BASIC_SCHEMA
      })

      expect(wrapper.exists()).toBe(true)
    })
  })

  describe('Special Characters', () => {
    it('should handle special characters in string fields', async () => {
      const specialChars = 'Test "quotes" & <tags> \\ backslash'
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: '' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', specialChars)
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.name).toBe(specialChars)
    })

    it('should handle unicode characters', async () => {
      const unicode = '中文測試 🚀 émojis'
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: '' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', unicode)
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.name).toBe(unicode)
    })

    it('should handle empty strings in fields', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', '')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.name).toBe('')
    })
  })

  describe('Number Field Edge Cases', () => {
    it('should handle zero as valid number', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '0')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.id).toBe(0)
    })

    it('should handle negative numbers', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '-42')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.id).toBe(-42)
    })

    it('should handle decimal numbers', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '3.14')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.id).toBe(3.14)
    })

    it('should handle very large numbers', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '999999999')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.id).toBe(999999999)
    })
  })

  describe('Date Field Edge Cases', () => {
    const DATE_SCHEMA = [
      { key: 'id', label: 'ID', type: 'number' as const },
      { key: 'date', label: 'Date', type: 'date' as const }
    ]

    it('should handle valid date format', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, date: '2025-01-15' }),
        schema: DATE_SCHEMA
      })

      const dateInput = wrapper.find('input#form-date') as any
      expect(dateInput.element.value).toBe('2025-01-15')
    })

    it('should update date field correctly', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, date: '2025-01-01' }),
        schema: DATE_SCHEMA
      })

      await fillInputField(wrapper, 'date', '2025-12-31')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.date).toBe('2025-12-31')
    })

    it('should handle empty date field', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, date: '' }),
        schema: DATE_SCHEMA
      })

      const dateInput = wrapper.find('input#form-date') as any
      expect(dateInput.element.value).toBe('')
    })
  })

  describe('Array Mode Edge Cases', () => {
    it('should handle adding item to empty array', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await fillInputField(wrapper, 'id', '1')
      await fillInputField(wrapper, 'name', 'First')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed).toHaveLength(1)
    })

    it('should handle deleting last item', async () => {
      window.confirm = vi.fn(() => true)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([{ id: 1, name: 'Only' }]),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'delete')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed).toEqual([])
    })

    it('should handle search with no matches', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([
          { id: 1, name: 'Alice' },
          { id: 2, name: 'Bob' }
        ]),
        schema: BASIC_SCHEMA
      })

      const searchInput = wrapper.find('input[placeholder*="Search"]')
      await searchInput.setValue('NonExistent')
      await nextTick()

      expect(wrapper.text()).toContain('No matching items found')
    })

    it('should handle inserting at first position', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([
          { id: 1, name: 'First' },
          { id: 2, name: 'Second' }
        ]),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'insert')
      await fillInputField(wrapper, 'id', '0')
      await fillInputField(wrapper, 'name', 'New First')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed[0]).toEqual({ id: 0, name: 'New First' })
      expect(parsed[1]).toEqual({ id: 1, name: 'First' })
    })
  })

  describe('Object Mode Edge Cases', () => {
    it('should handle multiple edits before save', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'First Edit')
      await fillInputField(wrapper, 'name', 'Second Edit')
      await fillInputField(wrapper, 'name', 'Final Edit')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = getEmittedModelValue(wrapper)
      const parsed = JSON.parse(emitted!)
      expect(parsed.name).toBe('Final Edit')
    })

    it('should handle cancel after multiple edits', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Edit 1')
      await fillInputField(wrapper, 'id', '999')
      await fillInputField(wrapper, 'name', 'Edit 2')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      const nameInput = wrapper.find('input#form-name') as any
      const idInput = wrapper.find('input#form-id') as any

      expect(nameInput.element.value).toBe('Original')
      expect(idInput.element.value).toBe('1')
    })

    it('should handle save-cancel-save cycle', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'First Save')
      await clickButton(wrapper, 'Save')
      await nextTick()

      await fillInputField(wrapper, 'name', 'Should Cancel')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      await fillInputField(wrapper, 'name', 'Second Save')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      const lastEmitted = emitted![emitted!.length - 1][0] as string
      const parsed = JSON.parse(lastEmitted)
      expect(parsed.name).toBe('Second Save')
    })
  })

  describe('Schema Mismatch', () => {
    it('should handle data with missing schema fields', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1 }),
        schema: BASIC_SCHEMA
      })

      const nameInput = wrapper.find('input#form-name')
      expect(nameInput.exists()).toBe(true)
    })

    it('should handle completely mismatched data structure', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({
          completelyDifferent: 'field',
          nothing: 'matches'
        }),
        schema: BASIC_SCHEMA
      })

      const idInput = wrapper.find('input#form-id')
      const nameInput = wrapper.find('input#form-name')

      expect(idInput.exists()).toBe(true)
      expect(nameInput.exists()).toBe(true)
      expect(idInput.attributes('type')).toBe('number')
      expect(nameInput.attributes('type')).toBe('text')
    })
  })

  describe('Confirm Dialog Cancellation', () => {
    it('should not delete when confirm returns false', async () => {
      window.confirm = vi.fn(() => false)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([{ id: 1, name: 'Keep Me' }]),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'delete')
      await nextTick()

      expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    })

    it('should not delete all when confirm returns false', async () => {
      window.confirm = vi.fn(() => false)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([
          { id: 1, name: 'A' },
          { id: 2, name: 'B' }
        ]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Delete All')
      await nextTick()

      expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    })
  })
})
