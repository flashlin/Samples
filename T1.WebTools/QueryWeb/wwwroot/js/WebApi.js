async function postJsonAsync(url, data) {
    const resp = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
            //Authentication: 'secret'
        },
        body: data == null ? '{}' : JSON.stringify(data)
    });
    return resp.json();
}