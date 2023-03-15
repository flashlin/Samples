async function postJsonAsync(url, data) {
    const resp = await fetch(url, {
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
            //Authentication: 'secret'
        },
        body: data == null ? '{}' : JSON.stringify(data)
    });
    return resp.json();
}

function postFormAsync(url, data) {
    const formData = new FormData();
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const value = data[key];
            formData.append(key, value);
        }
    }
    return fetch(url, {
        method: 'POST',
        body: formData,
    });
}