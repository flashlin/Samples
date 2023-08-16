import initSqlJs, { type Database } from 'sql.js';
import wasmPath from './assets/sql-wasm.wasm?url';

(async () => {
  //const wasmPath = import.meta.env.BASE_URL + 'assets/sql-wasm.wasm';
  //const SQL = await initSqlJs({ locateFile: () => wasmPath });
  const SQL = await initSqlJs({ locateFile: () => wasmPath });

  const db: Database = new SQL.Database();

  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY,
      name TEXT
    )
  `);

  // 插入數據
  const name = 'John Doe';
  db.exec(`INSERT INTO users (name) VALUES ('${name}')`);

  // 查詢數據
  const results = db.exec('SELECT * FROM users');
  const users = results[0].values;

  // 輸出結果
  console.log('users', users);
})();
