// @ts-ignore
// Import Hello function from linqjs
import { Hello } from 'linqjs';

// 範例客戶資料
const myCustomers = [
  { FirstName: "Chris", LastName: "Pearson", id: 1, status: "active" },
  { FirstName: "Kate", LastName: "Johnson", id: 2, status: "inactive" },
  { FirstName: "Josh", LastName: "Sutherland", id: 3, status: "active" },
  { FirstName: "John", LastName: "Ronald", id: 4, status: "inactive" },
  { FirstName: "Steve", LastName: "Pinkerton", id: 5, status: "active" }
];

// 測試訂單資料
const myOrders = [
  { OrderId: 101, CustomerId: 1, Amount: 250, Status: "shipped" },
  { OrderId: 102, CustomerId: 2, Amount: 120, Status: "pending" },
  { OrderId: 103, CustomerId: 1, Amount: 320, Status: "shipped" },
  { OrderId: 104, CustomerId: 3, Amount: 150, Status: "cancelled" },
  { OrderId: 105, CustomerId: 5, Amount: 500, Status: "shipped" }
];

// Demo: use Hello function
function runDemo() {
  const message = Hello('World');
  console.log(message);
}

runDemo(); 