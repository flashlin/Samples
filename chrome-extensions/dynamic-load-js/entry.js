const baseUrl = "http://localhost:8002/";
fetch(`${baseUrl}manifest.json`)
  .then((response) => response.json())
  .then((data) => {
    const indexHtml = data["index.html"];
    indexHtml.css.forEach((cssUrl) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = baseUrl + cssUrl;
      document.head.appendChild(link);
    });
    const script = document.createElement("script");
    script.src = baseUrl + indexHtml.file;
    document.body.appendChild(script);
  })
  .catch((error) => {
    console.error("Error downloading or parsing manifest.json:", error);
  });
