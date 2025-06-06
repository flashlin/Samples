//import { Buffer } from 'buffer';


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

// export function base64ToBlob(base64String: string) {
//     const byteCharacters = Buffer.from(base64String, 'base64').toString('binary');
//     const byteArrays = [];
//     for (let offset = 0; offset < byteCharacters.length; offset += 512) {
//         const slice = byteCharacters.slice(offset, offset + 512);
//         const byteNumbers = new Array(slice.length);
//         for (let i = 0; i < slice.length; i++) {
//             byteNumbers[i] = slice.charCodeAt(i);
//         }
//         const byteArray = new Uint8Array(byteNumbers);
//         byteArrays.push(byteArray);
//     }
//     return new Blob(byteArrays);
// }

export function base64ToBlob(base64String: string): Blob {
    const binaryString = atob(base64String);
    const length = binaryString.length;
    const uint8Array = new Uint8Array(length);
    for (let i = 0; i < length; i++) {
        uint8Array[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([uint8Array], { type: 'application/octet-stream' });
    return blob;
}