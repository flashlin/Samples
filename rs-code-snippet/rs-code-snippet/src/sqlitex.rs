mod Sqlitex {
   pub struct SqliteDb;

   impl SqliteDb {
      _connection: sqlite::Connection;

      //
      _connection = sqlite::open(":memory:").unwrap();

      pub fn init(& self) {
         //self.drive();
         self._connection.execute("
           CREATE TABLE users (name TEXT, age INTEGER);
           INSERT INTO users VALUES ('Alice', 42);
           INSERT INTO users VALUES ('Bob', 69);
         ").unwrap();
      }

      pub fn dump(& self) {
         self._connection.iterate("SELECT * FROM users WHERE age > 50", |pairs| {
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