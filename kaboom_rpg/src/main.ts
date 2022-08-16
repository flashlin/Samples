import kaboom from 'kaboom';
import { MapSize } from "./types";

kaboom();


const showMap = () => {
   for(let y = 0; y < MapSize.height; y++) {
      for(let x = 0; x < MapSize.width; x++) {
         console.log(`${x},${y}`);
      }
   }
};


onClick(() => {
   //addKaboom(mousePos())
})