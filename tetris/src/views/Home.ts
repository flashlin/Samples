import { Component, Vue } from 'vue-property-decorator';
import GameRect from '@/components/GameRect.vue'; // @ is an alias to /src
import { MinoType } from '@/models/Tetromino';
import { IState } from '@/stores/state';
import { State } from 'vuex-class';

@Component({
   components: {
      GameRect
   }
})
export default class Home extends Vue {
   @State((state: IState) => state.gameRect) rect!: MinoType[][];

   mounted() {
      //console.log('Home', this.rect);
   }
}
