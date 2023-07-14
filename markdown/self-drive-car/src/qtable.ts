export class QTable 
{
   qTable: Record<string, number> = {};
   epsilon = 0.1;

   constructor(numState: number, actions: number){
   }

   selectAction(currentState: number[]) {
      const randomValue = Math.random();
      // 如果随机值小于等于探索率，则进行探索，选择一个随机动作
      if (randomValue <= epsilon) {
         const randomAction = Math.floor(Math.random() * this.qTable.length);
         return randomAction;
      } else {
    // 否则，根据当前 Q 表格选择具有最大 Q 值的动作
    const stateIndex = getStateIndex(currentState);
    const stateActions = qTable[stateIndex];
    const maxQValue = Math.max(...stateActions);
    const maxActions = [];
    for (let i = 0; i < stateActions.length; i++) {
      if (stateActions[i] === maxQValue) {
        maxActions.push(i);
      }
    }
    // 从具有最大 Q 值的动作中随机选择一个
    const randomMaxAction = maxActions[Math.floor(Math.random() * maxActions.length)];
    return randomMaxAction;
      }
   }

   getQTableLength(): number {
      const size = Object.keys(this.qTable).length;
      return size;
   }
}
