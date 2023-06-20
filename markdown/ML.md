身分證驗證碼


```javascript {cmd="node"}
function convertIdToNumbers(id) {
  let a = id.substr(0, 1).charCodeAt() - 'A'.charCodeAt() + 10;
  let remainder = id.substr(1);
  let numberText = a + remainder;
  return numberText;
}

function calculateChecksum(id) {
  let weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1];
  let sum = 0;
  let idNumber = convertIdToNumbers(id)

  for (var i = 0; i < 10; i++) {
    num = idNumber.substr(i, 1);
    n = num.charCodeAt() - '0'.charCodeAt();
    sum += n * weights[i];
  }
  let checksum = 10 - sum % 10;
  return checksum;
}

var id = "A123456789";
var checksum = calculateChecksum(id);
console.log(checksum);
```
