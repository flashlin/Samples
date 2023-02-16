import { type ComponentPublicInstance, getCurrentInstance, inject, type InjectionKey, type Plugin, readonly } from 'vue'
import type { ILogger, LoggerOptions } from '@/utils/logger'
import { LoggerFactory } from '@/utils/logger'

const LOGGER_INJECTION_KEY: InjectionKey<ILogger> = Symbol('logger')
let activeLogger: ILogger

interface UseLoggerOptions extends LoggerOptions {
  captureComponentErrors?: boolean
}

function getErrorHandler( logger: ILogger ) {
  return ( error: unknown, vm: ComponentPublicInstance | null, info: string ) => {
    const tempError = error as Record<string, any>
    tempError.lifecycleHook = info
    if (vm && vm.$options) {
      const options = vm.$options
      tempError.component = vm.$root === vm
        ? 'Root'
        : options.name || options._componentTag || 'Anonymous'
      tempError.file = options.__file || ''
    }
    logger.logError('COMPONENT_ERROR', tempError)
  }
}

export function createLogger( options: UseLoggerOptions ): Plugin & { logger: ILogger } {
  const logger = readonly(LoggerFactory.createLogger(options))
  const { captureComponentErrors = true } = options
  activeLogger = logger

  return {
    install( app ) {
      app.provide(LOGGER_INJECTION_KEY, logger)
      if (captureComponentErrors) {
        app.config.errorHandler = getErrorHandler(logger)
      }
    },
    logger,
  }
}

export function useLogger(): ILogger {
  const logger = (getCurrentInstance() && inject(LOGGER_INJECTION_KEY)) || activeLogger
  if (!logger) {
    throw new Error("logger instance not found")
  }

  return logger
}