import { type App, getCurrentInstance, inject } from "vue"
import { isNullOrUndefined } from '@/utils/shared'

const APP_STATE_KEY = Symbol("APP_STATE_KEY")
type PropsType = Record<PropertyKey, boolean | number | string>

class DatasetProps<T> {
  state: T

  constructor( mountId: string, defaultProps: PropsType ) {
    const entryElement = document.getElementById(mountId)
    if (!entryElement) {
      throw new Error("entry point element not found")
    }

    this.state = this.convertDatasetToProps(entryElement.dataset, defaultProps) as unknown as T
    if (import.meta.env.DEV) {
      console.table(this.state)
    }
    Object.keys(entryElement.dataset).forEach(key => {
      delete entryElement.dataset[key]
    })
  }

  private convertDatasetToProps( dataset: DOMStringMap, defaultProps: PropsType ): PropsType {
    const props: PropsType = {}

    function isStringBoolean( datasetValue: string ): boolean {
      const valueToUpper = datasetValue.toUpperCase()
      return valueToUpper === "TRUE" || valueToUpper === "FALSE"
    }

    Object.keys(defaultProps).forEach(key => {
      const defaultPropsValue = defaultProps[key]
      const datasetValue = dataset[key]
      if (isNullOrUndefined(datasetValue)) {
        if (import.meta.env.DEV) {
          console.warn(`${ key } not found in dataset, use default value: ${ defaultPropsValue }`)
        }
        props[key] = defaultPropsValue
      }
      else {
        switch (typeof defaultPropsValue) {
          case "string":
            props[key] = datasetValue || defaultProps[key]
            break
          case 'number':
            props[key] = parseInt(datasetValue)
            break
          case 'boolean':
            props[key] = isStringBoolean(datasetValue)
              ? datasetValue.toUpperCase() === 'TRUE'
              : defaultPropsValue
            break
          default:
            props[key] = defaultPropsValue
        }
      }
    })
    return props
  }
}

export const createDatasetProps = <T extends { [key in keyof T]: T[key] }>( mountId: string, defaultValue: T ) => {
  if (!mountId || mountId.length === 0) {
    throw new Error("mountElement is required")
  }
  if (mountId.startsWith("#")) {
    mountId = mountId.substring(1)
  }

  const store = new DatasetProps<T>(mountId, defaultValue)
  return {
    install( app: App ) {
      app.provide(APP_STATE_KEY, store)
    },
    state: store.state,
  }
}

export const useEntryProps = <T extends { [key in keyof T]: T[key] }>() => {
  const currentInstance = getCurrentInstance()
  if (!currentInstance) {
    throw new Error("currentInstance not found")
  }
  const instance = inject<DatasetProps<T>>(APP_STATE_KEY)
  if (!instance) {
    throw new Error("entry props instance not provided")
  }
  return instance.state
}

