GM_addStyle(`
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
  }
  `);

function handleOnClick() {
    console.log('click 123');
}

const sidebarDiv = document.createElement('div');
sidebarDiv.className = 'sidebar';
sidebarDiv.onclick = handleOnClick;
document.body.appendChild(sidebarDiv);
console.log('Flash');