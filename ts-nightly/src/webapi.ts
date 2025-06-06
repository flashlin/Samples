import axios, { AxiosInstance, CreateAxiosDefaults } from 'axios';

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

    async postImageAsync<T>(url: string, filename: string, file: Blob): Promise<T> {
        const formData = new FormData();
        formData.append(filename, file);
        const resp = await axios.post(url, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        //const resp = await this._axiosInstance.post(url, data, { responseType: 'arraybuffer' });
        // const contentType = resp.headers['content-type'];
        // const blob = new Blob([resp.data], { type: contentType });
        // return blob;
        return resp.data;
    }

    postVoidAsync(url: string, data: any): Promise<void> {
       return this._axiosInstance.post(url, data);
    }

    async getAsync<T>(url: string, data: any): Promise<T> {
        const resp = await this._axiosInstance.get(url, { params: data });
        return resp.data;
    }

    getVoidAsync(url: string, data: any): Promise<void> {
        return this._axiosInstance.get(url, { params: data });
    }

    async getImageAsync(url: string, data: any): Promise<Blob> {
        const resp = await this._axiosInstance.get(url, {
            params: data,
            responseType: 'arraybuffer' });
        const contentType = resp.headers['content-type'];
        const blob = new Blob([resp.data], { type: contentType });
        //const imageUrl = URL.createObjectURL(blob);
        return blob;
    }
}

