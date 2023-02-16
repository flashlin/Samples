import { createHttpClient } from '@/plugins/request'
import type {
  GetValidationCodeResponse,
  RegisterRequest,
  RegisterResponse,
  RegistrationConfigurationRequest,
  RegistrationConfigurationResponse,
  ValidateNameRequest,
  ValidateNameResponse,
} from '@/pages/register/api/types'
import { getHost } from '@/utils/shared'


function getRegisterApiBaseURL() {
  return `${ window.location.protocol }//${ import.meta.env.VITE_AKIS_API_SUBDOMAIN }.${ getHost() }`
}

const httpClient = createHttpClient({
  baseURL: getRegisterApiBaseURL(),
  timeout: 20000,
})

export const fetchRegisterInitialInfoAsync = ( data: RegistrationConfigurationRequest ): Promise<RegistrationConfigurationResponse> =>
  httpClient.post<RegistrationConfigurationResponse>({
    url: 'api/Register/GetRegistrationConfiguration',
    data,
  })

export const validateNameAsync = ( data: ValidateNameRequest ): Promise<ValidateNameResponse> =>
  httpClient.post<ValidateNameResponse>({
    url: 'api/Register/ValidateName',
    data,
  })

export const getValidationCodeAsync = ( token: string ): Promise<GetValidationCodeResponse> =>
  httpClient.post<GetValidationCodeResponse>({
    url: 'api/Register/GetValidationCode',
    data: {
      token,
    },
  })

export const sendRegisterAsync = ( data: RegisterRequest ): Promise<RegisterResponse> =>
  httpClient.post<RegisterResponse>({
    url: 'api/Register/Create',
    data,
  })