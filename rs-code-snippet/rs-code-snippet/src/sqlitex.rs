//use sqlite::Error as SqError;

pub mod Sqlitex {
   

   //#[derive(Default)]
   //#[derive(Debug)]
   pub struct SqliteDb {
     name: String,
     _conn: sqlite::Connection,
   }
   
   impl SqliteDb {

      pub fn new(name: &String) -> SqliteDb {
         let obj = SqliteDb {
            name: name.to_string(),
            _conn: sqlite::Connection::open(name).unwrap(),
         };
         obj.init();
         return obj;
      }
      
      pub fn init(& self) {
         self._conn.execute("
           CREATE TABLE users (name TEXT, age INTEGER);
           INSERT INTO users VALUES ('Alice', 42);
           INSERT INTO users VALUES ('Bob', 69);
         ").unwrap();
      }

      pub fn dump(& self) {
         self._conn.iterate("SELECT * FROM users WHERE age > 50", |pairs| {
            for &(column, value) in pairs.iter() {
               println!("{} = {}", column, value.unwrap());
            }
            true
         }).unwrap();
      }
   }

   impl SqliteDb {
      fn drive(& self) {
         println!("Driving a car...");
      }
   }
}