import type { EventBusKey } from '@vueuse/core'

export const SetLoginNameUnavailableEvent: EventBusKey<{ suggestLists: string[] }> = Symbol()
export const TriggerPasswordValidateEvent: EventBusKey<{}> = Symbol()
export const TriggerValidationCodeErrorEvent: EventBusKey<{}> = Symbol()
export const TriggerReloadValidationCodeEvent: EventBusKey<{}> = Symbol()
export const SetPromotionCodeInvalidEvent: EventBusKey<{}> = Symbol()