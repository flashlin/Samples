import axios from "axios";

export class JwtApi {
   async get<T>(url: string): Promise<T> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios
         .get(url, 
            { 
               headers: {"Authorization" : `Bearer ${accessToken}`} 
            });
      return resp.data;
   }
}

export default new JwtApi();
