//import { M_addStyle } from "./tampermonkey/monkey";
//const css = ``;
//M_addStyle(css);

// import { catchAnyAnchor, clearAnchorAttributes } from "./helpers/htmlHelper";
// const anchorInfo = catchAnyAnchor();
// clearAnchorAttributes(anchorInfo.elem);
// export const appDiv = anchorInfo.elem;

export const appDiv = document.createElement('div');
document.body.appendChild(appDiv);
