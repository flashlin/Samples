import { M_addStyle } from "./tampermonkey/monkey";

const css = `
  .flash-sidebar {
    margin: 0;
    padding: 0;
    width: 30px;
    height: 30px;
    background-color: #04AA6D;
    overflow: auto;
    position: fixed;
    left: 0;
    top: 0;
    z-index: 9999;
  }
  
  .flash-sidebar expand {
    width: 410px;
    height: 510px;
  }
  
  .flash-sidebar a {
    display: block;
    color: black;
    padding: 16px;
    text-decoration: none;
  }
   
  .flash-sidebar a.active {
    background-color: #04AA6D;
    color: white;
  }
  
  .flash-sidebar a:hover:not(.active) {
    background-color: #555;
    color: white;
  }
  
  div.flash-sidebar-content {
    margin-left: 0px;
    padding: 0px 30px;
    width: 100%;
    height: 100%;
    background-color: #515251;
    position: absolute;
    left: 0;
    top: 0;
    display: none;
    z-index: 9999;
  }
`;

M_addStyle(css);

export const contentDiv = document.createElement('div');
contentDiv.className = 'flash-sidebar-content';
document.body.appendChild(contentDiv);

function handleOnClick() {
  console.log(`display='${contentDiv.style.display}'`);
  if (contentDiv.style.display == "block") {
    contentDiv.style.display = "none";
  } else {
    contentDiv.style.display = "block";
  }
}

const sidebarDiv = document.createElement('div');
sidebarDiv.className = 'flash-sidebar';
sidebarDiv.onclick = handleOnClick;
document.body.appendChild(sidebarDiv);
console.log('Flash');
