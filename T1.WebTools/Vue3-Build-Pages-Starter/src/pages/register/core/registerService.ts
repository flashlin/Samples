import type {
  RegisterErrorResponse,
  RegisterRequest,
  RegisterResponse,
  RegisterSuccessfullyResponse,
} from '@/pages/register/api/types'
import { sendRegisterAsync } from '@/pages/register/api/registerApiClient'

export class RegisterService {
  async createAccount( registerRequest: RegisterRequest, successCallBack: ( response: RegisterSuccessfullyResponse, registerRequest: RegisterRequest ) => void, errorCallBack: ( response: RegisterErrorResponse ) => void ) {
    try {
      const response = await sendRegisterAsync(registerRequest)
      this.isResponseSuccess(response)
        ? successCallBack(response, registerRequest)
        : errorCallBack(response)
    }
    catch (e) {
      console.log(e)
      window.alert('Error!')
    }
  }
  private isResponseSuccess(response: RegisterResponse): response is RegisterSuccessfullyResponse{
    return response.success
  }
}