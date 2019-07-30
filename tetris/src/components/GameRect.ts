import { Component, Vue } from 'vue-property-decorator';
import { MinoType } from '@/models/Tetromino';

@Component({
   components: {},
   props: {
      rect: Array
   }
})
export default class GameRect extends Vue {
   rect!: MinoType[][];

   flatRect: MinoBoxRow[] = [];

   mounted() {
      for (let y = 0; y < this.rect.length; y++) {
         let row = new MinoBoxRow();
         row.rowId = y;
         for (let x = 0; x < this.rect[y].length; x++) {
            let b = new MinoBox();
            b.columnId = x;
            b.box = this.rect[y][x];
            b.klass = b.box == MinoType.None ? '' : 'c';
            row.cols.push(b);
         }
         this.flatRect.push(row);
      }
   }
}

class MinoBoxRow {
   rowId: number = 0;
   cols: MinoBox[] = [];
}

class MinoBox {
   columnId: number = 0;
   klass: string = '';
   box: MinoType = MinoType.None;
}
