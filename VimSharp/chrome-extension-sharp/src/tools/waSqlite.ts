// 假設你使用 wa-sqlite/dist/wa-sqlite.mjs 作為模組入口
import SQLiteESMFactory from 'wa-sqlite/dist/wa-sqlite.mjs'
import * as SQLite from 'wa-sqlite'

export async function hello() {
  // 載入 WASM 模組
  const module = await SQLiteESMFactory()
  // 建立 SQLite 工廠
  const sqlite3 = SQLite.Factory(module)
  // 開啟資料庫（可指定 VFS，例如 'myDB', 'file:myDB?vfs=IDBBatchAtomicVFS' 等）
  const db = await sqlite3.open_v2('myDB')
  // 執行 SQL
  await sqlite3.exec(db, `CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)`)
  await sqlite3.exec(db, `INSERT INTO test (name) VALUES ('Vue3')`)
  // 查詢資料
  await sqlite3.exec(db, `SELECT * FROM test`, (row, _columns) => {
    console.log(row)
  })
  // 關閉資料庫
  await sqlite3.close(db)
}
