import { rest } from 'msw';

interface LoginRequestBody {
   username: string
   password: string
}

interface LoginResponseBody {
   token: string
}

export const handlers: any = [
   rest.post<LoginRequestBody, LoginResponseBody>(
      '/api/v1/auth/login',
      (req, res, ctx) => {
         return res(
            ctx.json({
               token: Date.now()
            })
         )
      }),
]   