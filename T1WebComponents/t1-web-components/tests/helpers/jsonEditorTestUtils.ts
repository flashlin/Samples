import { mount, VueWrapper } from '@vue/test-utils'
import JsonEditor from '@/components/JsonEditor.vue'
import type { JsonSchemaField } from '@/components/JsonEditor.vue'

export interface MountOptions {
  modelValue?: string | null
  schema: JsonSchemaField[]
  compact?: boolean
}

export function mountJsonEditor(options: MountOptions): VueWrapper {
  return mount(JsonEditor, {
    props: {
      modelValue: options.modelValue ?? null,
      schema: options.schema,
      compact: options.compact ?? false
    }
  })
}

export async function fillInputField(
  wrapper: VueWrapper,
  fieldKey: string,
  value: string
): Promise<void> {
  const input = wrapper.find(`input#${fieldKey}`)
  if (!input.exists()) {
    const formInput = wrapper.find(`input#form-${fieldKey}`)
    if (formInput.exists()) {
      await formInput.setValue(value)
      await formInput.trigger('input')
      return
    }
    throw new Error(`Input field "${fieldKey}" not found`)
  }
  await input.setValue(value)
  await input.trigger('input')
}

export async function clickButton(
  wrapper: VueWrapper,
  buttonText: string
): Promise<void> {
  const button = wrapper.findAll('button').find(btn =>
    btn.text().includes(buttonText)
  )
  if (!button) throw new Error(`Button "${buttonText}" not found`)
  await button.trigger('click')
}

export async function clickRowActionButton(
  wrapper: VueWrapper,
  rowIndex: number,
  actionType: 'edit' | 'delete' | 'insert'
): Promise<void> {
  const rows = wrapper.findAll('tbody tr')
  const row = rows[rowIndex]
  if (!row) throw new Error(`Row ${rowIndex} not found`)

  const buttons = row.findAll('button')
  let targetButton: typeof buttons[0] | undefined

  const actionTextMap = {
    edit: 'Edit',
    delete: 'Delete',
    insert: 'Add Before'
  }

  targetButton = buttons.find(btn => btn.text().includes(actionTextMap[actionType]))

  if (!targetButton) throw new Error(`Button ${actionType} not found in row ${rowIndex}`)
  await targetButton.trigger('click')
}

export function getEmittedModelValue(wrapper: VueWrapper): string | null {
  const emitted = wrapper.emitted('update:modelValue')
  if (!emitted || emitted.length === 0) return null
  return emitted[emitted.length - 1][0] as string
}

export function parseEmittedJson<T>(wrapper: VueWrapper): T {
  const value = getEmittedModelValue(wrapper)
  if (!value) throw new Error('No modelValue emitted')
  return JSON.parse(value)
}

export async function clickMainSaveButton(wrapper: VueWrapper): Promise<void> {
  const buttons = wrapper.findAll('button')
  const saveButtons = buttons.filter(btn => btn.text().includes('Save'))

  // Find Save button that is NOT in the modal (modal Save has bg-blue-600)
  const mainSaveButton = saveButtons.find(btn => {
    const classes = btn.classes()
    return !classes.includes('bg-blue-600')
  })

  if (!mainSaveButton) {
    throw new Error('Main Save button not found')
  }

  await mainSaveButton.trigger('click')
}
