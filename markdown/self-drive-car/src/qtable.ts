export class QTable 
{
   qTable: Record<string, number[]> = {};
   epsilon = 0.1;
   numActions = 0;
   learningRate = 0.1;
   discountFactor = 0.9;   //折扣因子

   constructor(numState: number, numActions: number){
      this.numActions = numActions;
   }

   selectAction(currentState: number[]) {
      const randomValue = Math.random();
      // 如果随机值小于等于探索率，则进行探索，选择一个随机动作
      if (randomValue <= this.epsilon) {
         const randomAction = Math.floor(Math.random() * this.getQTableLength());
         return randomAction;
      }
      // 否则，根据当前 Q 表格选择具有最大 Q 值的动作
      const stateIndex = this.getStateIndex(currentState);
      const stateActions = this.getQTableStateActions(stateIndex);
      const maxQValue = Math.max(...stateActions);
      const maxQActions = [];
      for (let i = 0; i < stateActions.length; i++) {
         if (stateActions[i] === maxQValue) {
            maxQActions.push(i);
         }
      }
      // 从具有最大 Q 值的动作中随机选择一个
      const randomMaxAction = maxQActions[Math.floor(Math.random() * maxQActions.length)];
      return randomMaxAction;
   }

   updateQValue(currentState: number[], action: number, reward: number, nextState: number[]): void {
      const currentStateIndex = this.getStateIndex(currentState);
      const nextStateIndex = this.getStateIndex(nextState);
    
      const currentStateActions = this.getQTableStateActions(currentStateIndex);
      const currentQValue = currentStateActions[action];
      const maxNextQValue = Math.max(...this.getQTableStateActions(nextStateIndex));
      const updatedQValue = currentQValue + this.learningRate * (reward + this.discountFactor * maxNextQValue - currentQValue);
 
      currentStateActions[action] = updatedQValue;
   }

   getQTableStateActions(stateIndex: string) {
      if( !this.qTable.hasOwnProperty(stateIndex) ) {
         this.qTable[stateIndex] = [];
         for (let i = 0; i < this.numActions; i++) {
            this.qTable[stateIndex].push(0);
         }
      }
      const stateActions = this.qTable[stateIndex];
      return stateActions;
   }

   getStateIndex(state: number[]) {
      const index = state.join('-');
      return index;
   }

   getQTableLength(): number {
      const size = Object.keys(this.qTable).length;
      return size;
   }
}
