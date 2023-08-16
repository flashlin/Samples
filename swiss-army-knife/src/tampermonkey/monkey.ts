export let M_addStyle = (content: string) => {
    const styleElem = document.createElement('style');
    styleElem.innerHTML = content;
    document.body.appendChild(styleElem);
    return styleElem;
}

if (process.env.NODE_ENV != 'development') {
    M_addStyle = GM_addStyle;
}