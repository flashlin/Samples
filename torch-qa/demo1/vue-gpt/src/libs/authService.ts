import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL + '/api/v1/auth/';

export interface ILoginReq {
   loginName: string;
   password: string;
}

export interface ILoginResp {
   token: string;
}

class AuthService {
   accessToken: string = '';
   handle: number = 0;

   async login(req: ILoginReq) {
      const resp = await axios
         .post(API_URL + 'login', {
            loginName: req.loginName,
            password: req.password
         });
      this.saveAccessToken(resp.data.token);
      this.clearHandle();
      this.handle = window.setInterval(()=> {
         this.refreshAccessToken();
      }, 1000 * 60);
      return resp.data as ILoginResp;
   }

   logout() {
      this.clearHandle();
      this.accessToken = '';
      localStorage.removeItem('accessToken');
   }

   clearHandle() {
      if( this.handle != 0) {
         window.clearInterval(this.handle);
         this.handle = 0;
      }
   }

   async refreshAccessToken() {
      const accessToken = localStorage.getItem('accessToken');
      if( accessToken === '' ) {
         return;
      }
      try {
         const resp = await axios
            .post(API_URL + 'refreshToken', {},
               {
                  headers: { "Authorization": `Bearer ${accessToken}` }
               });
         this.saveAccessToken(resp.data.token);
         return resp.data;
      } catch {
         this.saveAccessToken('');
      }
   }

   saveAccessToken(token: string) {
      this.accessToken = token;
      localStorage.setItem('accessToken', token);
   }
}

export default new AuthService();