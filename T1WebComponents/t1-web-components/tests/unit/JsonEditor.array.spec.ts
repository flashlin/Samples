import { describe, it, expect, vi } from 'vitest'
import { nextTick } from 'vue'
import {
  mountJsonEditor,
  clickButton,
  clickRowActionButton,
  fillInputField,
  parseEmittedJson,
  getEmittedModelValue,
  clickModalSaveButton,
  clickMainSaveButton
} from '../helpers/jsonEditorTestUtils'
import {
  BASIC_SCHEMA,
  FULL_SCHEMA,
  SAMPLE_ARRAY_DATA,
  createArrayJson
} from '../helpers/mockData'

describe('JsonEditor - Array Mode', () => {
  describe('Initial Rendering', () => {
    it('should render table with correct headers from schema', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(true)
      expect(wrapper.text()).toContain('ID')
      expect(wrapper.text()).toContain('Name')
    })

    it('should display all rows from JSON array', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      const rows = wrapper.findAll('tbody tr')
      expect(rows.length).toBeGreaterThanOrEqual(3)
      expect(wrapper.text()).toContain('Alice')
      expect(wrapper.text()).toContain('Bob')
      expect(wrapper.text()).toContain('Charlie')
    })

    it('should show empty message when array is empty', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.text()).toContain('No matching items found')
    })
  })

  describe('Search Functionality', () => {
    it('should filter rows based on search query', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      const searchInput = wrapper.find('input[placeholder*="Search"]')
      await searchInput.setValue('Alice')
      await nextTick()

      const bodyText = wrapper.find('tbody').text()
      expect(bodyText).toContain('Alice')
      expect(bodyText).not.toContain('Bob')
      expect(bodyText).not.toContain('Charlie')
    })

    it('should search across all columns', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      const searchInput = wrapper.find('input[placeholder*="Search"]')

      await searchInput.setValue('2')
      await nextTick()
      let bodyText = wrapper.find('tbody').text()
      expect(bodyText).toContain('Bob')

      await searchInput.setValue('bob')
      await nextTick()
      bodyText = wrapper.find('tbody').text()
      expect(bodyText).toContain('Bob')
    })

    it('should be case insensitive', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      const searchInput = wrapper.find('input[placeholder*="Search"]')
      await searchInput.setValue('ALICE')
      await nextTick()

      const bodyText = wrapper.find('tbody').text()
      expect(bodyText).toContain('Alice')
    })

    it('should show all rows when search is cleared', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      const searchInput = wrapper.find('input[placeholder*="Search"]')
      await searchInput.setValue('Alice')
      await nextTick()
      await searchInput.setValue('')
      await nextTick()

      const bodyText = wrapper.find('tbody').text()
      expect(bodyText).toContain('Alice')
      expect(bodyText).toContain('Bob')
      expect(bodyText).toContain('Charlie')
    })
  })

  describe('Add New Item', () => {
    it('should open add dialog when Add button clicked', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await nextTick()

      expect(wrapper.text()).toContain('Add New Item')
    })

    it('should add new item to end of array', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await fillInputField(wrapper, 'id', '4')
      await fillInputField(wrapper, 'name', 'David')
      await clickModalSaveButton(wrapper)
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(result).toHaveLength(4)
      expect(result[3]).toEqual({ id: 4, name: 'David' })
    })

    it('should initialize fields with correct default values', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await nextTick()

      const idInput = wrapper.find('input#id')
      const nameInput = wrapper.find('input#name')

      expect((idInput.element as HTMLInputElement).value).toBe('0')
      expect((nameInput.element as HTMLInputElement).value).toBe('')
    })

    it('should not emit update when cancelled', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await fillInputField(wrapper, 'name', 'Should Not Save')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeUndefined()
    })
  })

  describe('Insert Item', () => {
    it('should insert item before specified row', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 1, 'insert')
      await fillInputField(wrapper, 'id', '99')
      await fillInputField(wrapper, 'name', 'Inserted')
      await clickModalSaveButton(wrapper)
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(result).toHaveLength(4)
      expect(result[1]).toEqual({ id: 99, name: 'Inserted' })
      expect(result[2]).toEqual({ id: 2, name: 'Bob' })
    })

    it('should show correct dialog title for insert', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'insert')
      await nextTick()

      expect(wrapper.text()).toContain('Add Item Before')
    })
  })

  describe('Edit Item', () => {
    it('should populate dialog with existing item data', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'edit')
      await nextTick()

      const idInput = wrapper.find('input#id') as any
      const nameInput = wrapper.find('input#name') as any

      expect(idInput.element.value).toBe('1')
      expect(nameInput.element.value).toBe('Alice')
    })

    it('should update item when saved', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 1, 'edit')
      await fillInputField(wrapper, 'name', 'Bob Updated')
      await clickModalSaveButton(wrapper)
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(result[1]).toEqual({ id: 2, name: 'Bob Updated' })
    })

    it('should not change item when cancelled', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'edit')
      await fillInputField(wrapper, 'name', 'Should Not Change')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeUndefined()
    })
  })

  describe('Delete Item', () => {
    it('should delete item after confirmation', async () => {
      window.confirm = vi.fn(() => true)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 1, 'delete')
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(result).toHaveLength(2)
      expect(result.find(item => item.id === 2)).toBeUndefined()
    })

    it('should not delete when cancelled', async () => {
      window.confirm = vi.fn(() => false)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 1, 'delete')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeUndefined()
    })

    it('should show confirmation dialog with correct message', async () => {
      const confirmSpy = vi.fn(() => true)
      window.confirm = confirmSpy

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'delete')

      expect(confirmSpy).toHaveBeenCalledWith(
        expect.stringContaining('delete')
      )
    })
  })

  describe('Delete All', () => {
    it('should delete all items after confirmation', async () => {
      window.confirm = vi.fn(() => true)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Delete All')
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(result).toEqual([])
    })

    it('should not delete when cancelled', async () => {
      window.confirm = vi.fn(() => false)

      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_ARRAY_DATA),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Delete All')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeUndefined()
    })

    it('should hide Delete All button when array is empty', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      const deleteAllBtn = wrapper.findAll('button').find(btn =>
        btn.text().includes('Delete All')
      )
      expect(deleteAllBtn).toBeUndefined()
    })
  })

  describe('Field Type Handling', () => {
    it('should render number input for number type', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await nextTick()

      const idInput = wrapper.find('input#id')
      expect(idInput.attributes('type')).toBe('number')
    })

    it('should render date input for date type', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: FULL_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await nextTick()

      const dateInput = wrapper.find('input#createdAt')
      expect(dateInput.attributes('type')).toBe('date')
    })

    it('should render text input for string type', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await nextTick()

      const nameInput = wrapper.find('input#name')
      expect(nameInput.attributes('type')).toBe('text')
    })

    it('should convert number input to number in output', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA
      })

      await clickButton(wrapper, 'Add')
      await fillInputField(wrapper, 'id', '42')
      await fillInputField(wrapper, 'name', 'Test')
      await clickModalSaveButton(wrapper)
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(result[0].id).toBe(42)
      expect(typeof result[0].id).toBe('number')
    })
  })

  describe('Compact Output', () => {
    it('should emit compact JSON when compact=true', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA,
        compact: true
      })

      await clickButton(wrapper, 'Add')
      await fillInputField(wrapper, 'id', '1')
      await fillInputField(wrapper, 'name', 'Test')
      await clickModalSaveButton(wrapper)
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const emittedValue = getEmittedModelValue(wrapper)
      expect(emittedValue).not.toContain('\n')
      expect(emittedValue).not.toContain('  ')
    })

    it('should emit formatted JSON when compact=false', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([]),
        schema: BASIC_SCHEMA,
        compact: false
      })

      await clickButton(wrapper, 'Add')
      await fillInputField(wrapper, 'id', '1')
      await fillInputField(wrapper, 'name', 'Test')
      await clickModalSaveButton(wrapper)
      await nextTick()

      // Click main Save button to persist changes
      await clickMainSaveButton(wrapper)
      await nextTick()

      const emittedValue = getEmittedModelValue(wrapper)
      expect(emittedValue).toContain('\n')
      expect(emittedValue).toContain('  ')
    })
  })
})
