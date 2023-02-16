import type { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosRequestHeaders, AxiosResponse } from 'axios'
import Axios from 'axios'

export function createAxiosInstance( config?: AxiosRequestConfig, responseErrorHandler?: ( error: AxiosError ) => void ): AxiosInstance {
  const defaultOptions = {
    baseURL: window.location.origin,
    timeout: 5000,
    withCredentials: true,
  }
  const axiosInstance = Axios.create({ ...defaultOptions, ...config })
  axiosInstance.interceptors.response.use(
    ( res ) => {
      return res
    },
    ( error: AxiosError ) => {
      if (responseErrorHandler) {
        responseErrorHandler(error)
      }
      return Promise.reject(error)
    },
  )
  return axiosInstance
}

interface RequestConfig<D = any> {
  url: string
  data?: D
  params?: any
  headers?: AxiosRequestHeaders
  timeout?: number
}

export interface IRequestProxy {
  post<T>(req: RequestConfig ): Promise<T>
  get<T>(req: RequestConfig ): Promise<T>
}

export class RequestProxy implements IRequestProxy {
  private _axiosInstance: AxiosInstance

  constructor( axiosInstance: AxiosInstance ) {
    this._axiosInstance = axiosInstance
  }

  post<T>( { url, data, timeout, headers }: RequestConfig ): Promise<T> {
    return this._axiosInstance
      .post(url, data, { timeout, headers })
      .then(( res: AxiosResponse<T> ) => {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // ts-expect-error
        if (res.data === '' || res.data == null) {
          return res as unknown as T
        }
        return res.data
      })
  }

  get<T>( { url, timeout, params, headers }: RequestConfig ): Promise<T> {
    return this._axiosInstance
      .get(url, { params, timeout, headers })
      .then(( res: AxiosResponse<T> ) => res.data)
  }
}

export const createHttpClient = ( config?: AxiosRequestConfig, responseErrorHandler?: ( error: AxiosError ) => void ): IRequestProxy => new RequestProxy(createAxiosInstance(config, responseErrorHandler))
export default new RequestProxy(createAxiosInstance())

