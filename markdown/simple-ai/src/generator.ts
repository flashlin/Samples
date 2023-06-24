function convertIdToNumbers(id: string) {
    const first: any = {
        'A': 10,
        'B': 11,
        'C': 12,
        'D': 13,
        'E': 14,
        'F': 15,
        'G': 16,
        'H': 17,
        'I': 34,
        'J': 18,
        'K': 19,
        'L': 20,
        'M': 21,
        'N': 22,
        'O': 35,
        'P': 23,
        'Q': 24,
        'R': 25,
        'S': 26,
        'U': 28,
        'V': 29,
        'W': 32,
        'X': 30,
        'Y': 31,
        'Z': 33,
    }
    let a = first[id.substring(0, 1)];
    let remainder = id.substring(1);
    let numberText = `${a}` + remainder;
    return numberText;
}

function calculateChecksum(idNumbers: string) {
    let weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let sum = 0;

    for (var i = 0; i < 10; i++) {
        let num = idNumbers.substring(i, i + 1);
        let n = num.charCodeAt(0) - '0'.charCodeAt(0);
        sum += n * weights[i];
    }
    let checksum = 10 - sum % 10;
    if( checksum == 10 ) {
        return 0;
    }
    return checksum;
}

export function generateRandomID() {
    var firstChar = String.fromCharCode(65 + Math.floor(Math.random() * 26));

    var secondNum = Math.floor(Math.random() * 2) + 1;

    let nums = '';
    for (var i = 0; i < 7; i++) {
        let num = Math.floor(Math.random() * 10);
        nums += `${num}`;
    }

    // 計算檢查碼
    let str = `${firstChar}${secondNum}${nums}`;
    let idNumbers = convertIdToNumbers(str);
    let checkCode = calculateChecksum(idNumbers);

    //console.log(`${str} ${checkCode}`);

    let id = str + checkCode;
    return id;
}


export function getInputs(id: string): number[] {
  const idStr = id.substring(0, 9);
  const inputs = convertIdToNumbers(idStr);
  const numbers = [];
  for(let len=0; len<inputs.length; len++) {
    numbers.push(inputs[len].charCodeAt(0) - '0'.charCodeAt(0));
  }
  return numbers;
}

function generateItem() {
  const id = generateRandomID();
  const inputs = getInputs(id);
  const target = id.substring(9).charCodeAt(0) - '0'.charCodeAt(0);
  return { inputs, target };
}

export function generateTrainData(dataSize: number) {
  let xTrain = [];
  let yTrain = [];
  for (let n = 0; n < dataSize; n++) {
    const item = generateItem();
    xTrain.push(item.inputs);
    yTrain.push(item.target);
  }
  return { xTrain, yTrain };
}