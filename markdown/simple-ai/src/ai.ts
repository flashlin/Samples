export function program(x: number) {
    console.log(x)
    return x;
}


export function convertIdToNumbers(idStr: string) {
    let numbers = [];
    let a = idStr.substring(0, 1).charCodeAt(0) - 'A'.charCodeAt(0) + 10;
    numbers.push(a);
    //console.log('first', idStr.substr(0, 1), a);
  
    for(let i=1; i<idStr.length; i++) {
      let numStr = idStr.substring(i, 1);
      let num = numStr.charCodeAt(0) - '0'.charCodeAt(0);
      numbers.push(num);
    }
    return numbers;
  }