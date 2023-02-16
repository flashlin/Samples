import { useI18n } from 'vue-i18n'

export const useNotification = () => {
  const { t } = useI18n()
  return {
    alert: ( key: string, custom: Record<string, unknown> = {} ): void => {
      window.alert(t(key, custom))
    },
  }
}