import type { ILoginReq, ILoginResp } from '@/libs/authService';
import { rest } from 'msw';


export const handlers: any = [
   rest.post<ILoginReq, ILoginResp>(
      '/api/v1/auth/login',
      (req, res, ctx) => {
         const resp: ILoginResp = {
            token: Date.now().toString()
         };
         return res(
            ctx.json(resp)
         )
      }),
]   