export function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
  
      reader.onload = () => {
        const buffer = reader.result as ArrayBuffer;
        const base64 = arrayBufferToBase64(buffer);
        resolve(base64);
      };
  
      reader.onerror = () => {
        reject(new Error('File read error'));
      };
  
      reader.readAsArrayBuffer(file);
    });
  }
  
  function arrayBufferToBase64(buffer: ArrayBuffer): string {
    const binary = Array.from(new Uint8Array(buffer))
      .map(byte => String.fromCharCode(byte))
      .join('');
  
    return btoa(binary);
  }