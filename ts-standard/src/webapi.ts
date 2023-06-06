import axios, { AxiosInstance, CreateAxiosDefaults, HeadersDefaults } from 'axios';

export class WebApi {
    private _axiosInstance: AxiosInstance;

    constructor(options?: Partial<CreateAxiosDefaults<any>>) {
        const apiOptions = Object.assign({
            timeout: 5000,
            headers: {
                'Content-Type': 'application/json',
            },
        }, options);
        this._axiosInstance = axios.create(apiOptions);
    }

    async postAsync<T>(url: string, data: any): Promise<T> {
        const resp = await this._axiosInstance.post(url, data);
        return resp.data;
    }

    async postImageAsync(url: string, data: any): Promise<Blob> {
        const resp = await this._axiosInstance.post(url, data, { responseType: 'blob' });
        return resp.data;
    }

    async postVoidAsync(url: string, data: any): Promise<void> {
        await this._axiosInstance.post(url, data);
    }
}

