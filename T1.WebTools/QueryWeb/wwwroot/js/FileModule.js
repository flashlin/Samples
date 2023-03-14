function uploadChunkAsync(uploadUrl, file, chunkSize, currentChunk, totalChunks) {
    const start = currentChunk * chunkSize;
    const end = Math.min(file.size, start + chunkSize);
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsArrayBuffer(file.slice(start, end));
        reader.onload = async function () {
            try{
                await postJsonAsync(uploadUrl, {
                    chunk: reader.result,
                    currentChunk: currentChunk,
                    totalChunks: totalChunks,
                });
                resolve();
            } catch (e1){
                reject(e1);
            }
        }
    })
}

async function uploadFileAsync(uploadUrl, file) {
    const chunkSize = 1024 * 2;
    const totalChunks = Math.ceil(file.size / chunkSize);
    for await (let currentChunk = 0; currentChunk < totalChunks; currentChunk++) {
        await uploadChunkAsync(file, currentChunk, totalChunks, chunkSize);
    }
}

async function uploadFileElementAsync(uploadUrl, fileElement) {
    //const uploadUrl = "https://example.com/upload";
    //const fileElement = document.getElementById("fileInput");
    for await (let i =0; i<fileElement.files.length; i++)
    {
        const file = fileElement.files[i];
        await uploadFileAsync(uploadUrl, file);
    }
}