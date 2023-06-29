function writeLog(message) {
  const outputDiv = document.getElementById("output");
  outputDiv.innerHTML = outputDiv.innerHTML + message + "<br>";
}

function generateId9() {
  let firstChar = String.fromCharCode(65 + Math.floor(Math.random() * 26));
  let secondNum = Math.floor(Math.random() * 2) + 1;
  let nums = "";
  for (let i = 0; i < 7; i++) {
    let num = Math.floor(Math.random() * 10);
    nums += `${num}`;
  }
  let id9 = `${firstChar}${secondNum}${nums}`;
  return id9;
}

function convertIdToNumbersList(id) {
  const first = {
    A: 10,
    B: 11,
    C: 12,
    D: 13,
    E: 14,
    F: 15,
    G: 16,
    H: 17,
    I: 34,
    J: 18,
    K: 19,
    L: 20,
    M: 21,
    N: 22,
    O: 35,
    P: 23,
    Q: 24,
    R: 25,
    S: 26,
    T: 27,
    U: 28,
    V: 29,
    W: 32,
    X: 30,
    Y: 31,
    Z: 33,
  };

  let result = [];
  for (let i = 0; i < id.length; i++) {
    if (i == 0) {
      result.push(first[id.substring(0, 1)]);
      continue;
    }
    let num = id.substring(i, i + 1);
    let n = num.charCodeAt(0) - "0".charCodeAt(0);
    result.push(n);
  }

  return result;
}

function calculateChecksum(idNumbersList) {
  let weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1];
  let sum = 0;
  for (var i = 0; i < idNumbersList.length; i++) {
    let num = idNumbersList.slice(i, i + 1)[0];
    sum += num * weights[i];
  }
  let checksum = 10 - (sum % 10);
  if (checksum == 10) {
    return 0;
  }
  return checksum;
}

function generateRandomID() {
  let id9 = generateId9();
  let idNumbers = convertIdToNumbersList(id9);
  let checkCode = calculateChecksum(idNumbers);
  let id = `${id9}${checkCode}`;
  return id;
}

function generateTrainItem() {
  let id9 = generateId9();

  let first = id9.substring(0, 1).charCodeAt(0) - "A".charCodeAt(0) + 10;
  let trainNumbers = [first];
  for (let n = 1; n < id9.length; n++) {
    let num = id9.substring(n, n + 1).charCodeAt(0) - "0".charCodeAt(0);
    trainNumbers.push(num);
  }

  let idNumbers = convertIdToNumbersList(id9);
  let checkCode = calculateChecksum(idNumbers);
  
  //writeLog(`${id9} ${trainNumbers} ${checkCode}`);
  return [trainNumbers, checkCode];
}

function generateTrainData(dataSize) {
  let xTrain = [];
  let yTrain = [];
  for (let n = 0; n < dataSize; n++) {
    let [x, y] = generateTrainItem();
    xTrain.push(x);
    yTrain.push(y);
  }
  return [xTrain, yTrain];
}
