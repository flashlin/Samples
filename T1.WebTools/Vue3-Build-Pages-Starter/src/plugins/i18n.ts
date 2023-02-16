import { type App, getCurrentInstance, inject } from 'vue'
import { createI18n as createClientI18n, type DefineDateTimeFormat, type I18n } from 'vue-i18n'
import { LanguageType } from '@/constants/language'

type LocaleFile = Record<string, any>
export type LanguageResources = Record<string, () => Promise<any>>
type Messages = { [key in LanguageType]?: LocaleFile }
export interface IResourceLoader {
  getResourceAsync( language: LanguageType): Promise<LocaleFile>
}

export class ResourceLoader implements IResourceLoader {
  private readonly resources: LanguageResources

  constructor( resources: LanguageResources = {} ) {
    this.resources = resources
  }

  async getResourceAsync( language: string ): Promise<LocaleFile> {
    const searchKey = language.toUpperCase().replace('_', '-')
    let resource = {}

    for (const key of Object.keys(this.resources)) {
      if (key.toUpperCase().includes(searchKey)) {
        try {
          const res = await this.resources[key]()
          resource = { ...resource, ...res.default }
        }
        catch (e) {
          console.log('load locale file failed', e)
        }
      }
    }

    return resource
  }
}

export class I18nFactory {
  static async createDefault( resources: LanguageResources, locale: LanguageType = LanguageType.EN ) {
    const resourceLoader = new ResourceLoader(resources)
    const messages: Messages = {}
    messages[LanguageType.EN] = await resourceLoader.getResourceAsync(LanguageType.EN)

    if (locale !== LanguageType.EN) {
      messages[locale] = await resourceLoader.getResourceAsync(locale)
    }
    return createClientI18n({
      legacy: false,
      globalInjection: true,
      locale,
      fallbackLocale: LanguageType.EN,
      messages,
    })
  }

  static async create( defaultResource: LocaleFile, resources: LanguageResources, locale: LanguageType = LanguageType.EN ) {
    const messages: Messages = {}
    messages[LanguageType.EN] = defaultResource
    const resourceLoader = new ResourceLoader(resources)

    if (locale !== LanguageType.EN) {
      messages[locale] = await resourceLoader.getResourceAsync(locale)
    }
    return createClientI18n({
      legacy: false,
      globalInjection: true,
      locale,
      fallbackLocale: LanguageType.EN,
      messages,
    })
  }
}

const RESOURCE_MANAGEMENT_KEY = Symbol("RESOURCE_MANAGEMENT_KEY")

export const useResourceManagement = () => {
  const currentInstance = getCurrentInstance()
  if (!currentInstance) {
    throw new Error("currentInstance not found")
  }
  const instance = inject<IResourceManagement>(RESOURCE_MANAGEMENT_KEY)
  if (!instance) {
    throw new Error("ResourceManagement instance not provided")
  }
  return instance
}

interface I18nManagement {
  readonly i18n: I18n
  readonly management: IResourceManagement

  install( app: App, ...options: unknown[] ): void
}

export interface IResourceManagement {
  updateLanguageAsync( language: LanguageType | IntlLanguageType ): Promise<void>
}

export type DateTimeFormats<LangType extends keyof any> = { [key in LangType]?: DefineDateTimeFormat }

interface CreateResourceManagementOptions {
  defaultResource: LocaleFile
  resources: LanguageResources
  locale: LanguageType
  datetimeFormats?: DateTimeFormats<LanguageType>
}
export class ResourceManagementFactory {
  static async createAsync( {
    defaultResource,
    resources,
    locale,
    datetimeFormats,
  }: CreateResourceManagementOptions ): Promise<I18nManagement> {
    const resourceLoader = new ResourceLoader(resources)
    const messages: Messages = {}
    messages[LanguageType.EN] = defaultResource

    if (locale !== LanguageType.EN) {
      messages[locale] = await resourceLoader.getResourceAsync(locale)
    }

    const i18n: I18n = createClientI18n({
      legacy: false,
      globalInjection: true,
      locale,
      fallbackLocale: LanguageType.EN,
      datetimeFormats,
      messages,
    })

    const resourceManagement = new ResourceManagement({
      i18n,
      resourceLoader,
    })

    return {
      management: resourceManagement,
      i18n,
      install: ( app ) => {
        app.use(i18n)
        app.provide(RESOURCE_MANAGEMENT_KEY, resourceManagement)
      },
    }
  }
}

interface ResourceManagementOptions {
  i18n: I18n
  resourceLoader: IResourceLoader
}

export class ResourceManagement implements IResourceManagement {
  private readonly i18n: I18n
  private resourceLoader: IResourceLoader

  constructor( { i18n, resourceLoader }: ResourceManagementOptions ) {
    this.i18n = i18n
    this.resourceLoader = resourceLoader
  }

  async updateLanguageAsync( language: LanguageType | IntlLanguageType ): Promise<void> {
    if (!this.i18n.global.availableLocales.includes(language)) {
      const languageResource = await this.resourceLoader.getResourceAsync(language)
      this.i18n.global.setLocaleMessage(language, languageResource)
    }
    // @ts-expect-error global locale is ref type
    this.i18n.global.locale.value = language
  }
}