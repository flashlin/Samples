export function patrol(distance=50, speed=20, dir=1) {
   return {
      id: 'patrol',
      require: ['pos', 'area', ],
      startingPos: vec2(0, 0),
      add() {
         this.startingPos = this.pos;
         this.on('collide', (obj, side)=>{
            if( side === "left" || side === 'right') {
               dir = -dir;
            }
         });
      },
      update() {
         if( Math.abs(this.pos.x - this.startingPos.x) >= distance ) {
            dir = -dir;
         }
         this.move(speed * dir, 0);
      },
   };
}