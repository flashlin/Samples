import type { ILoginReq, ILoginResp } from '@/libs/authService';
import { rest } from 'msw';

function sleep(ms: number) {
   return new Promise(resolve => setTimeout(resolve, ms));
}

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


   rest.get('/api/v1/chat/conversation',
      async (req, res, ctx) => {
         const eventStreamData = `using System;
         public class Main {
            public void Run() {
               Console .Write ("HELLO") ;
            }
         }
         `;

         for (const eventData of eventStreamData.split(' ')) {
            await sleep(1000);
            res(
               ctx.status(200),
               ctx.set('Content-Type', 'text/event-stream'),
               ctx.body(`${eventData}\n\n`),
            );
         }

         res(
            ctx.status(200),
            ctx.set('Content-Type', 'text/event-stream'),
            ctx.body(`data: [DONE]`),
         );
      }),
]   