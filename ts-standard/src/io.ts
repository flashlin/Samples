export function blobToImageUrlData(blob: Blob): Promise<string> {
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