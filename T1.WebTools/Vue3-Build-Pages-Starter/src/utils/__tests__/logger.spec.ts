import { describe, expect } from 'vitest'
import {
  ApiLoggerDecorator,
  ConsoleLogger,
  EmptyLogger,
  type IClientLogProxy,
  type ILogger,
  LoggerFactory,
} from '../logger'
import { mock, type MockProxy } from 'vitest-mock-extended'
import { AxiosError } from 'axios'

describe('ApiLoggerDecorator', () => {
  const TEST_PAGE_NAME = 'test'
  let logger: MockProxy<ILogger>
  let clientLogProxy: MockProxy<IClientLogProxy>
  beforeEach(() => {
    vi.restoreAllMocks()
    logger = mock<ILogger>()
    clientLogProxy = mock<IClientLogProxy>()
    clientLogProxy.sendLogAsync.mockResolvedValue()
  })

  it('should log information and send elk information log', () => {
    const expectedMessage = 'test information message'

    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['information'],
    })
    sut.logInformation(expectedMessage)

    expect(logger.logInformation).toHaveBeenCalledWith(expectedMessage)
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalledWith(expect.objectContaining({
      message: expectedMessage,
      level: 'information',
      page: TEST_PAGE_NAME,
    }))
  })

  it('should log debug and send elk debug log', () => {
    const expectedMessage = 'test debug message'

    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['debug'],
    })
    sut.logDebug(expectedMessage)

    expect(logger.logDebug).toHaveBeenCalledWith(expectedMessage)
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalledWith(expect.objectContaining({
      message: expectedMessage,
      level: 'debug',
      page: TEST_PAGE_NAME,
    }))
  })


  it('should log error and send elk error log', () => {
    const expectedMessage = 'test error message'
    vi.spyOn(JSON, 'stringify').mockImplementation(( value ) => value)
    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['error'],
    })
    sut.logError(expectedMessage)

    expect(logger.logError).toHaveBeenCalledWith(expectedMessage, undefined)
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalledWith(expect.objectContaining({
      message: { message: expectedMessage },
      page: TEST_PAGE_NAME,
    }))
  })


  it('should log error and send elk error log', () => {
    const expectedMessage = 'test error message'
    vi.spyOn(JSON, 'stringify').mockImplementation(( value ) => value)
    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['error'],
    })
    sut.logError(expectedMessage)

    expect(logger.logError).toHaveBeenCalledWith(expectedMessage, undefined)
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.objectContaining({
        message: expectedMessage,
      }),
    }))
  })

  it('should get error properties when error is instance of Error', () => {
    const expectedMessage = 'test error message'
    const expectedErrorMessage = "throw_test_error"
    const expectedError = new Error(expectedErrorMessage)
    vi.spyOn(JSON, 'stringify').mockImplementation(( value ) => value)

    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['error'],
    })
    sut.logError(expectedMessage, expectedError)

    expect(logger.logError).toHaveBeenCalledWith(expectedMessage, expect.any(Error))
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.objectContaining({
        message: expectedMessage,
        exceptionMessage: expectedErrorMessage,
        exceptionStack: expect.any(String),
        exceptionType: expectedError.name,
      }),
    }))
  })

  it('should get error properties when error is unknown type', () => {
    const expectedMessage = 'test error message'
    const expectedError = new AxiosError()
    vi.spyOn(JSON, 'stringify').mockImplementation(( value ) => value)

    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['error'],
    })
    sut.logError(expectedMessage, expectedError)

    expect(logger.logError).toHaveBeenCalledWith(expectedMessage, expect.any(Error))
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.objectContaining({
        message: expectedMessage,
        exceptionType: expectedError.name,
      }),
    }))
  })

  it('should not send api when log level not allowed', () => {
    const expectedMessage = 'test error message'

    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: [],
    })
    sut.logInformation(expectedMessage)

    expect(logger.logInformation).toHaveBeenCalled()
    expect(clientLogProxy.sendLogAsync).not.toHaveBeenCalled()
  })

  it('should send information log when log level in allow list', () => {
    const expectedMessage = 'test error message'

    const sut = new ApiLoggerDecorator(logger, clientLogProxy, {
      pageName: TEST_PAGE_NAME,
      allowLevels: ['information'],
    })
    sut.logInformation(expectedMessage)

    expect(logger.logInformation).toHaveBeenCalled()
    expect(clientLogProxy.sendLogAsync).toHaveBeenCalled()
  })

})

describe('LoggerFactory', () => {
  it('should use console log when enableConsole is true', () => {
    const logger = LoggerFactory.createLogger({
      enableConsole: true,
      enableApi: false,
      pageName: 'test',
    })

    expect(logger).instanceof(ConsoleLogger)
  })


  it('should use empty log when enableConsole is false', () => {
    const logger = LoggerFactory.createLogger({
      enableConsole: false,
      enableApi: false,
      pageName: 'test',
    })

    expect(logger).instanceof(EmptyLogger)
  })

  it('should add api log decorator when enable api log is true', () => {
    const logger = LoggerFactory.createLogger({
      enableConsole: false,
      enableApi: true,
      pageName: 'test',
    })

    expect(logger).instanceof(ApiLoggerDecorator)
  })

  it('should add api log decorator when enable api log is false', () => {
    const logger = LoggerFactory.createLogger({
      enableConsole: false,
      enableApi: false,
      pageName: 'test',
    })

    expect(logger).instanceof(EmptyLogger)
  })
})