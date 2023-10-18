import axios, { type AxiosResponse } from "axios";

const API_URL = import.meta.env.VITE_API_URL;

export class JwtApi {
   async get<T>(url: string): Promise<T> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios
         .get(`${API_URL}${url}`, 
            { 
               headers: {"Authorization" : `Bearer ${accessToken}`} 
            });
      return resp.data;
   }

   async post<T>(url: string, data: any = {}): Promise<T> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios.post(`${API_URL}${url}`, 
         JSON.stringify(data), 
         {
            headers: {
               Authorization: `Bearer ${accessToken}`,
            },
         });
      return resp.data;
   }

   async postResponse(url: string, data: any = {}): Promise<AxiosResponse<any, any>> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios.post(`${API_URL}${url}`, 
         JSON.stringify(data), 
         {
            headers: {
               Authorization: `Bearer ${accessToken}`,
            },
         });
      return resp;
   }
}

export default new JwtApi();
