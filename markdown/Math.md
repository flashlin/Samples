@import "https://cdn.plot.ly/plotly-2.24.1.min.js"

半圓弧

```javascript {cmd=true element="<div id='chart'></div>"}
// 線段的座標
const x1 = -6;
const y1 = 1;
const x2 = -1;
const y2 = 2;


// 半徑和起始角度、結束角度 (以弧度表示)
const cx = 0;
const cy = 0;
const radius = 5;
const startAngle = 90 * Math.PI / 180;
const endAngle = 180 * Math.PI / 180;


// 繪製線段的數據
const lineData = [{
    x: [x1, x1, x2],
    y: [y1, y1, y2],
    mode: 'lines',
    line: {
        color: 'blue',
        width: 2
    },
    name: 'line' 
}];

const arcData = {
  x: [],
  y: [],
  mode: 'lines',
  line: {
    color: 'red',
    width: 2
  },
  name: 'arc' 
};

const originData = {
  x: [cx],
  y: [cy],
  mode: 'markers',
  marker: {
    color: 'green',
    size: 8,
    symbol: 'circle'
  },
  name: 'cx,cy'
};


const pointsData = {
  x: [-8],
  y: [cy],
  mode: 'markers',
  marker: {
    color: 'green',
    size: 8,
    symbol: 'circle'
  },
  name: 'x,cy'
};


const angleStep = Math.PI / 180; // 每度的弧度增量
// 生成圓弧上的點的座標
for (let angle = startAngle; angle <= endAngle; angle += angleStep) {
  const x = cx + radius * Math.cos(angle);
  const y = cy + radius * Math.sin(angle);
  arcData.x.push(x);
  arcData.y.push(y);
}


const data = [...lineData, arcData, originData, pointsData];

addPoints([])
function addPoints(posList) {
    for(let i=0; i<posList.length; i++) {
        const { x, y, name } = posList[i];
        data.push({
            x: [x],
            y: [y],
            mode: 'markers',
            marker: {
                color: 'green',
                size: 8,
                symbol: 'circle'
            },
            name: name
        });
    }
}


// 設置格線的數據
const layout = {
    xaxis: {
        range: [-10, 10],
        showgrid: true,
        gridcolor: 'lightgray',
        dtick: 1
    },
    yaxis: {
        range: [-10, 10],
        showgrid: true,
        gridcolor: 'lightgray',
        dtick: 1
    },
    annotations: [
    {
      x: x1,
      y: y1,
      text: 'x1,y1',
      showarrow: false,
      font: {
        color: 'black'
      }
    },
    {
      x: x2,
      y: y2,
      text: 'x2,y2',
      showarrow: false,
      font: {
        color: 'black'
      }
    },
    {
      x: -radius,
      y: cy,
      text: '-r,cy',
      showarrow: false,
      font: {
        color: 'black'
      }
    },
    {
      x: cx,
      y: radius,
      text: 'cx,r',
      showarrow: false,
      font: {
        color: 'black'
      }
    },
    {
      x: cx,
      y: cy,
      text: 'cx,cy',
      showarrow: false,
      font: {
        color: 'black'
      }
    }
  ]
};


Plotly.newPlot('chart', data, layout);
```

線段斜率
\[ m = \frac{{y_2 - y_1}}{{x_2 - x_1}} \]



x2 = (y2 - cy) / m + cx
y2 = m * (x2 - cx) + cy