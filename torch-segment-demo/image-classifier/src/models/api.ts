import { WebApi, base64ToBlob, blobToImageUrl } from "ts-standard";
import type { IImageForClassify, IImageForClassifyData, IImageSegmentationReq, IImageSegmentationResp, ILabel } from "./types";

const client = new WebApi();


export async function getImageForClassifier(): Promise<IImageForClassify> {
    const resp = await client.getAsync<IImageForClassifyData>("/api/imageForClassifier", null);
    const imageBlob = base64ToBlob(resp.imageData);
    return {
        name: resp.name,
        imageUrl: await blobToImageUrl(imageBlob)
    };
}

export function getClassifyCategories(): Promise<ILabel[]> {
    return client.getAsync<ILabel[]>("/api/getClassifyCategories", null);
}

export function sendClassifyImage(id: number, imageName: string):  Promise<void> {
    const data = {
        id,
        imageName
    }
    return client.postVoidAsync('/api/classifyImage', data);
}

export function sendImageSegmentation(req: IImageSegmentationReq): Promise<IImageSegmentationResp> {
    return client.postImageAsync<IImageSegmentationResp>('/api/imageSegmentation', req);
}