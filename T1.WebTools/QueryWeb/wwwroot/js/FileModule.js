function blobSlice(blob, start, length)
{
    if(blob.slice){
        return blob.slice(start, length);
    }
    if(blob.webkitSlice){
        return blob.webkitSlice(start, length);
    }
    if(blob.mozSlice){
        return blob.mozSlice(start, length);
    }
    return null;
}
function uploadChunkAsync(uploadUrl, file, chunkSize, currentChunk, totalChunks) {
    const start = currentChunk * chunkSize;
    const end = Math.min(file.size, start + chunkSize);
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async function () {
            let chunk = blobSlice(reader.result, start, end);
            chunk = new Uint8Array(chunk);
            chunk = Array.from(chunk);
            try{
                await postFormAsync(uploadUrl, {
                    fileName: file.name,
                    chunk: chunk,
                    currentChunk: currentChunk,
                    totalChunks: totalChunks,
                });
                resolve();
            } catch (e1){
                reject(e1);
            }
        }
        reader.readAsArrayBuffer(file);
    })
}


async function uploadFileAsync(uploadUrl, file) {
    const chunkSize = 1024 * 2;
    const totalChunks = Math.ceil(file.size / chunkSize);
    console.log('uploadFileAsync', file, totalChunks);
    for (let currentChunk = 0; currentChunk < totalChunks; currentChunk++) {
        await uploadChunkAsync(uploadUrl, file, currentChunk, totalChunks, chunkSize);
    }
}

async function uploadFileElementAsync(uploadUrl, fileElement){
    for (let i=0; i<fileElement.files.length; i++)
    {
        const file = fileElement.files[i];
        await uploadFileAsync(uploadUrl, file);
    }
}
