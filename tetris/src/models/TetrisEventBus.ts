import t1 from 't1-scripts';
import { TetrisGameEvent } from './TetrisGameEvent';
export class TetrisEventBus extends t1.RxEventBus<TetrisGameEvent> {}
