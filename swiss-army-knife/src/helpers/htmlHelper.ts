export interface IAnchorInfo {
    attributes: { [key: string]: string };
    innerHTML: string;
    elem: HTMLAnchorElement;
}

export function getAnchorInfo(anchor: HTMLAnchorElement): IAnchorInfo {
    const attributes: { [key: string]: string } = {};
    for (let i = 0; i < anchor.attributes.length; i++) {
        const attr = anchor.attributes[i];
        attributes[attr.name] = attr.value;
    }
    return {
        attributes,
        innerHTML: anchor.innerHTML,
        elem: anchor,
    };
}

export function clearAnchorAttributes(anchor: HTMLAnchorElement): void {
    const attributesToRemove = [...anchor.attributes];
    attributesToRemove.forEach(attribute => {
        anchor.removeAttribute(attribute.name);
    });
}


export function restoreAnchor(anchor: HTMLAnchorElement, info: IAnchorInfo): void {
    for (const attrName in info.attributes) {
        anchor.setAttribute(attrName, info.attributes[attrName]);
    }
    anchor.innerHTML = info.innerHTML;
}

export function catchAnyAnchor(): IAnchorInfo {
    const originalAnchor = document.querySelector('a') as HTMLAnchorElement;
    return getAnchorInfo(originalAnchor);
}