function blobSlice(blob, start, length) {
	if (blob.slice) {
		return blob.slice(start, length);
	}
	if (blob.webkitSlice) {
		return blob.webkitSlice(start, length);
	}
	if (blob.mozSlice) {
		return blob.mozSlice(start, length);
	}
	return null;
}

async function uploadChunkAsync(uploadUrl, file, chunkSize, currentChunk, totalChunks) {
	const start = currentChunk * chunkSize;
	const end = Math.min(file.size, start + chunkSize);
	const chunk = file.slice(start, end);
	await postFormAsync(uploadUrl, {
		fileName: file.name,
		chunk: chunk,
		currentChunk: currentChunk,
		totalChunks: totalChunks,
	});
}


async function uploadFileAsync(uploadUrl, file) {
	const chunkSize = 1024 * 2;
	const totalChunks = Math.ceil(file.size / chunkSize);
	for (let currentChunk = 0; currentChunk < totalChunks; currentChunk++) {
		await uploadChunkAsync(uploadUrl, file, chunkSize, currentChunk, totalChunks);
	}
}

async function uploadFileElementAsync(uploadUrl, fileElement) {
	for (let i = 0; i < fileElement.files.length; i++) {
		const file = fileElement.files[i];
		await uploadFileAsync(uploadUrl, file);
	}
}
