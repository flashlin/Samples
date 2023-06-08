import { Buffer } from 'buffer';

export function blobToImageUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const data = reader.result as string;
            resolve(data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

export function base64ToBlob(base64String: string) {
    const byteCharacters = Buffer.from(base64String, 'base64').toString('binary');
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