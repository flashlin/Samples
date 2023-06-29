
(function(l, r) { if (!l || l.getElementById('livereloadscript')) return; r = l.createElement('script'); r.async = 1; r.src = '//' + (self.location.host || 'localhost').split(':')[0] + ':35729/livereload.js?snipver=1'; r.id = 'livereloadscript'; l.getElementsByTagName('head')[0].appendChild(r) })(self.document);
(function () {
  'use strict';

  var img = "data:image/png;base64,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";

  const carWidth = 75;
  const carHeight = 117;
  const roadLength = 220;
  const roadWidth = 220;
  const roadColor = 'red';
  const roadMargin = 22;
  var RoadType;
  (function (RoadType) {
      RoadType[RoadType["Vertical"] = 0] = "Vertical";
      RoadType[RoadType["Horizontal"] = 1] = "Horizontal";
      RoadType[RoadType["LeftTopCurve"] = 2] = "LeftTopCurve";
      RoadType[RoadType["LeftBottomCurve"] = 3] = "LeftBottomCurve";
      RoadType[RoadType["RightTopCurve"] = 4] = "RightTopCurve";
      RoadType[RoadType["RightBottomCurve"] = 5] = "RightBottomCurve";
  })(RoadType || (RoadType = {}));
  //垂直
  class VerticalRoad {
      constructor(pos = { x: 0, y: 0 }) {
          this.pos = pos;
      }
      render(ctx) {
          const pos = this.pos;
          ctx.beginPath();
          ctx.moveTo(pos.x + roadMargin, pos.y);
          ctx.lineTo(pos.x + roadMargin, pos.y + roadLength);
          ctx.moveTo(pos.x + roadWidth - roadMargin, pos.y);
          ctx.lineTo(pos.x + roadWidth - roadMargin, pos.y + roadLength);
          ctx.lineWidth = 7;
          ctx.strokeStyle = roadColor; // 線條顏色
          ctx.stroke();
      }
  }
  // 水平
  class HorizontalRoad {
      constructor(pos = { x: 0, y: 0 }) {
          this.pos = pos;
      }
      render(ctx) {
          const pos = this.pos;
          ctx.beginPath();
          ctx.moveTo(pos.x, pos.y + roadMargin);
          ctx.lineTo(pos.x + roadLength, pos.y + roadMargin);
          ctx.moveTo(pos.x, pos.y + roadWidth - roadMargin);
          ctx.lineTo(pos.x + roadLength, pos.y + roadWidth - roadMargin);
          ctx.lineWidth = 7;
          ctx.strokeStyle = roadColor;
          ctx.stroke();
      }
  }
  class LeftTopCurve {
      constructor(pos = { x: 0, y: 0 }) {
          this.pos = pos;
      }
      render(ctx) {
          const pos = this.pos;
          pos.x += roadLength;
          pos.y += roadWidth;
          ctx.beginPath();
          //ctx.arc(pos.x, pos.y, roadWidth - roadMargin, Math.PI, 1.5 * Math.PI);
          ctx.arc(pos.x, pos.y, roadWidth - roadMargin, Math.PI, 1.5 * Math.PI);
          ctx.strokeStyle = "red";
          ctx.lineWidth = 7;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(pos.x + roadMargin - roadMargin, pos.y, roadMargin, Math.PI, 1.5 * Math.PI);
          ctx.strokeStyle = "red";
          ctx.lineWidth = 7;
          ctx.stroke();
      }
  }
  function create2dArray(width, height) {
      let arr = new Array(width);
      for (let i = 0; i < arr.length; i++) {
          arr[i] = new Array(height);
      }
      return arr;
  }
  class RoadMap {
      constructor(pos = { x: 0, y: 0 }) {
          this.roads = create2dArray(10, 10);
          this.pos = pos;
          this.roads[0][0] = new LeftTopCurve();
          this.roads[0][1] = new VerticalRoad();
          this.roads[1][0] = new HorizontalRoad();
      }
      render(ctx) {
          const pos = this.pos;
          const roads = this.roads;
          for (let ix = 0; ix < roads.length; ix++) {
              for (let iy = 0; iy < roads[ix].length; iy++) {
                  const road = roads[ix][iy];
                  if (road == null) {
                      continue;
                  }
                  road.pos.x = pos.x + ix * roadWidth;
                  road.pos.y = pos.y + iy * roadLength;
                  road.render(ctx);
              }
          }
      }
  }
  class Car {
      constructor(pos) {
          this.pos = pos;
          this.carImage = new Image();
          this.carImage.src = img;
      }
      render(ctx) {
          const pos = this.pos;
          //ctx.globalCompositeOperation = 'destination-atop';
          ctx.drawImage(this.carImage, pos.x, pos.y, carWidth, carHeight);
      }
  }
  class Game {
      constructor() {
          this.canvas = this.createCanvas();
          this.ctx = this.canvas.getContext("2d");
      }
      createCanvas() {
          //const canvas = document.createElement("canvas");
          //document.body.appendChild(canvas);
          const canvas = document.getElementById("canvas");
          canvas.width = 800;
          canvas.height = 600;
          return canvas;
      }
      drawF4Car() {
          const ctx = this.ctx;
          const car = new Car({ x: 23 + carWidth, y: roadLength });
          car.render(ctx);
          const car2 = new Car({ x: 23, y: roadLength });
          car2.render(ctx);
      }
      drawRoad() {
          const ctx = this.ctx;
          const roadMap = new RoadMap();
          roadMap.render(ctx);
          // const road = new HorizontalRoad({ x: 3 + roadWidth, y: 0 });
          // road.render(ctx);
          // const road1 = new LeftTopCurve({ x: 0, y: 0 });
          // road1.render(ctx);
          // const road2 = new VerticalRoad({ x: 0, y: roadLength });
          // road2.render(ctx);
      }
      render() {
          const ctx = this.ctx;
          const canvas = this.canvas;
          //ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          this.drawRoad();
          this.drawF4Car();
          requestAnimationFrame(this.render.bind(this));
      }
  }
  function main() {
      console.log('start...');
      const game = new Game();
      game.render();
  }
  main();

})();
