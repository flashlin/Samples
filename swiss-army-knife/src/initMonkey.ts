//import { M_addStyle } from "./tampermonkey/monkey";
//const css = ``;
//M_addStyle(css);

// import { catchAnyAnchor, clearAnchorAttributes } from "./helpers/htmlHelper";
// const anchorInfo = catchAnyAnchor();
// clearAnchorAttributes(anchorInfo.elem);
// export const appDiv = anchorInfo.elem;

const bodyElement = document.body;
export const appDiv = document.createElement('div');
//const firstChild = document.body.firstChild;
//bodyElement.insertBefore(appDiv, firstChild);
bodyElement.appendChild(appDiv);
