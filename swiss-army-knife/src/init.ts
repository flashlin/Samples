
const css = `
  .sidebar {
    margin: 0;
    padding: 0;
    width: 30px;
    height: 30px;
    background-color: #04AA6D;
    overflow: auto;
    position: fixed;
    left: 0;
    top: 0;
    z-index: 1000;
  }
  
  .sidebar expand {
    width: 410px;
    height: 510px;
  }
  
  .sidebar a {
    display: block;
    color: black;
    padding: 16px;
    text-decoration: none;
  }
   
  .sidebar a.active {
    background-color: #04AA6D;
    color: white;
  }
  
  .sidebar a:hover:not(.active) {
    background-color: #555;
    color: white;
  }
  
  div.sidebar-content {
    margin-left: 0px;
    padding: 51px 0px;
    height: 100%;
    background-color: #515251;
    display: none;
  }
`;

//GM_addStyle(css);
function addStyle(css: string) {
    const styleElem = document.createElement('style');
    styleElem.innerHTML = css;
    document.body.appendChild(styleElem);
}
addStyle(css);

export const contentDiv = document.createElement('div');
contentDiv.className = 'sidebar-content';
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
sidebarDiv.className = 'sidebar';
sidebarDiv.onclick = handleOnClick;
document.body.appendChild(sidebarDiv);
console.log('Flash');
