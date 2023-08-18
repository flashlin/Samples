

export function exportToCsv(name: string, data: any[]) {
    const csvContent = "data:text/csv;charset=utf-8," + data.map(item => Object.values(item).join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${name}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
}