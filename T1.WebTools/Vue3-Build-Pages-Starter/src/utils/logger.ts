import { createHttpClient, type IRequestProxy } from '@/plugins/request'
import { getHost } from '@/utils/shared'

interface LogContext {
  message: string
  level: string
  page: string
  host: string
  path: string
  queryString: string
  scheme: string
}

export interface IClientLogProxy {
  sendLogAsync( logContext: LogContext ): Promise<void>
}

interface ErrorContext {
  message: string
  exceptionStack?: string
  exceptionMessage?: string
  exceptionType?: string
}

export interface ILogger {
  logError( message: string, error?: unknown ): void

  logInformation( message: string ): void

  logDebug( message: string ): void
}

type LogLevels = 'error' | 'information' | 'debug'

export class ConsoleLogger implements ILogger {
  logError( message: string, error?: unknown ): void {
    console.error(message, error)
  }

  logInformation( message: string ): void {
    console.log(message)
  }

  logDebug( message: string ): void {
    console.debug(message)
  }
}

export class EmptyLogger implements ILogger {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  logError( message: string, error?: unknown ): void {
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  logInformation( message: string ): void {
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  logDebug( message: string ): void {
  }
}

type ApiLoggerOptions = Pick<LoggerOptions, 'allowLevels' | 'pageName'>

export class ApiLoggerDecorator implements ILogger {
  private _options: Required<ApiLoggerOptions>

  constructor( private _logger: ILogger, private _clientLogProxy: IClientLogProxy, options: ApiLoggerOptions ) {
    this._options = {
      pageName: options.pageName,
      allowLevels: options.allowLevels ?? ['error', 'information', 'debug'],
    }
  }

  logError( message: string, error?: unknown ) {
    this._logger.logError(message, error)
    const errorContext: ErrorContext = {
      message,
      ...this.getExceptionProperties(error),
      ...this.getOtherProperties(error),
    }
    this.sendLogRequest('error', JSON.stringify(errorContext))
  }

  logInformation( message: string ) {
    this._logger.logInformation(message)
    this.sendLogRequest('information', message)
  }

  logDebug( message: string ) {
    this._logger.logDebug(message)
    this.sendLogRequest('debug', message)
  }

  private getOtherProperties( error: unknown ): Record<string, any> {
    if (error && typeof error === 'object') {
      const IGNORE_KEYS = ['stack', 'message', 'name']
      let propertyFound = false
      const properties: Record<string, any> = {}
      Object.keys(error).forEach(( key ) => {
        if (IGNORE_KEYS.indexOf(key) >= 0) return

        let val = (error as Record<string, any>)[key]
        if (val == null || typeof val === 'function') return

        if (typeof val === 'object' && typeof val.toISOString === 'function') {
          val = val.toISOString()
        }

        properties[key] = val
        propertyFound = true
      })

      if (propertyFound) {
        return properties
      }
    }
    return {}
  }

  private getExceptionProperties( error: unknown ) {
    if (error && error instanceof Error) {
      const { message: exceptionMessage, stack: exceptionStack, name: exceptionType } = error
      return { exceptionMessage, exceptionStack, exceptionType }
    }
    return {}
  }

  private sendLogRequest( level: LogLevels, message: string ) {
    if (!this._options.allowLevels.includes(level)) return

    this._clientLogProxy.sendLogAsync({
      level,
      message,
      scheme: location.protocol.slice(0, -1),
      host: location.host,
      path: location.pathname,
      queryString: location.search,
      page: this._options.pageName,
    }).then().catch(e => console.error(e))
  }
}


class LogApiProxy implements IClientLogProxy {
  constructor( private httpClient: IRequestProxy ) {
  }

  async sendLogAsync( logContext: LogContext ): Promise<void> {
    return this.httpClient.post({
      url: '/api/track/ClientLog',
      data: logContext,
    })
  }
}

export interface LoggerOptions {
  pageName: string
  allowLevels?: LogLevels[]
  enableConsole: boolean
  enableApi: boolean
}

export class LoggerFactory {
  public static createLogger( options: LoggerOptions ): ILogger {
    const logger = options.enableConsole ? new ConsoleLogger() : new EmptyLogger()
    return !options.enableApi
      ? logger
      : new ApiLoggerDecorator(logger, this.createLogApiProxy(), options)
  }

  private static createLogApiProxy() {
    return new LogApiProxy(createHttpClient({
      baseURL: `${ location.protocol }//cmgw.${ getHost() }`,
    }))
  }
}