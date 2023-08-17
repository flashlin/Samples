import { M_addStyle } from "./tampermonkey/monkey";

const css = `
.flash-icon {
  display: inline-block;
  width: 32px;
  height: 32px;
  font-size: 24px;
  font-weight: bold;
  color: gold;
  text-align: center;
  line-height: 32px;
  background: none;
  cursor: pointer;
}

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
  z-index: 999;
  background: none;
  overflow: hidden;
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
  z-index: 998;
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
sidebarDiv.className = 'flash-sidebar flash-icon';
sidebarDiv.onclick = handleOnClick;
sidebarDiv.innerText = 'F';
document.body.appendChild(sidebarDiv);
console.log('Flash');
