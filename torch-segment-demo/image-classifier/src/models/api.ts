import { WebApi, blobToImageUrl } from "ts-standard";
import type { IImageForClassify, IImageForClassifyData, ILabel } from "./types";

const client = new WebApi();

function base64ToBlob(base64String: string) {
    const byteCharacters = atob(base64String);
    console.log('atob ok');
    const byteArrays = [];
    
    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
      const slice = byteCharacters.slice(offset, offset + 512);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
  
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays);
  }

export async function getImageForClassifier(): Promise<IImageForClassify> {
    const resp = await client.getAsync<IImageForClassifyData>("/api/imageForClassifier", null);
    console.log("get image", resp);
    const imageBlob = base64ToBlob(resp.imageData);
    console.log("get blob", imageBlob);
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
    return client.getVoidAsync('/api/classifyImage', data);
}

